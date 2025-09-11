# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from openfold.model.embedders import (
    InputEmbedder,
    RecyclingEmbedder,
    TemplateAngleEmbedder,
    TemplatePairEmbedder,
    ExtraMSAEmbedder,
)
from openfold.model.evoformer import EvoformerStack, ExtraMSAStack
from openfold.model.heads import AuxiliaryHeads
from openfold.model.structure_module import (
    StructureModule as OpenFoldStructureModule,
    StructureModuleTransition,
    BackboneUpdate,
    AngleResnet,
    InvariantPointAttention,
    torsion_angles_to_frames,
    frames_and_literature_positions_to_atom14_pos,
)
from openfold.model.template import (
    TemplatePairStack,
    TemplatePointwiseAttention,
    embed_templates_average,
    embed_templates_offload,
)
from openfold.model.primitives import LayerNorm, Linear
import openfold.np.residue_constants as residue_constants
from openfold.np.residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)
from openfold.utils.rigid_utils import Rigid, Rotation
from openfold.utils.feats import (
    pseudo_beta_fn,
    build_extra_msa_feat,
    build_template_angle_feat,
    build_template_pair_feat,
    atom14_to_atom37,
)
from openfold.utils.tensor_utils import (
    add,
    tensor_tree_map,
    dict_multimap,
)


class StructureModule(nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
        c_ipa,
        c_resnet,
        no_heads_ipa,
        no_qk_points,
        no_v_points,
        dropout_rate,
        no_blocks,
        no_transition_layers,
        no_resnet_blocks,
        no_angles,
        trans_scale_factor,
        epsilon,
        inf,
        **kwargs,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_ipa:
                IPA hidden channel dimension
            c_resnet:
                Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
            no_heads_ipa:
                Number of IPA heads
            no_qk_points:
                Number of query/key points to generate during IPA
            no_v_points:
                Number of value points to generate during IPA
            dropout_rate:
                Dropout rate used throughout the layer
            no_blocks:
                Number of structure module blocks
            no_transition_layers:
                Number of layers in the single representation transition
                (Alg. 23 lines 8-9)
            no_resnet_blocks:
                Number of blocks in the angle resnet
            no_angles:
                Number of angles to generate in the angle resnet
            trans_scale_factor:
                Scale of single representation transition hidden dimension
            epsilon:
                Small number used in angle resnet normalization
            inf:
                Large number used for attention masking
        """
        super(StructureModule, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_ipa = c_ipa
        self.c_resnet = c_resnet
        self.no_heads_ipa = no_heads_ipa
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.dropout_rate = dropout_rate
        self.no_blocks = no_blocks
        self.no_transition_layers = no_transition_layers
        self.no_resnet_blocks = no_resnet_blocks
        self.no_angles = no_angles
        self.trans_scale_factor = trans_scale_factor
        self.epsilon = epsilon
        self.inf = inf

        # Buffers to be lazily initialized later
        # self.default_frames
        # self.group_idx
        # self.atom_mask
        # self.lit_positions

        self.layer_norm_s = LayerNorm(self.c_s)
        self.layer_norm_z = LayerNorm(self.c_z)

        self.linear_in = Linear(self.c_s, self.c_s)

        self.ipa = InvariantPointAttention(
            self.c_s,
            self.c_z,
            self.c_ipa,
            self.no_heads_ipa,
            self.no_qk_points,
            self.no_v_points,
            inf=self.inf,
            eps=self.epsilon,
        )

        self.ipa_dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm_ipa = LayerNorm(self.c_s)

        self.transition = StructureModuleTransition(
            self.c_s,
            self.no_transition_layers,
            self.dropout_rate,
        )

        self.bb_update = BackboneUpdate(self.c_s)

        self.angle_resnet = AngleResnet(
            self.c_s,
            self.c_resnet,
            self.no_resnet_blocks,
            self.no_angles,
            self.epsilon,
        )

    def forward(
        self,
        evoformer_output_dict,
        aatype,
        mask=None,
        inplace_safe=False,
        _offload_inference=False,
    ):
        """
        Args:
            evoformer_output_dict:
                Dictionary containing:
                    "single":
                        [*, N_res, C_s] single representation
                    "pair":
                        [*, N_res, N_res, C_z] pair representation
            aatype:
                [*, N_res] amino acid indices
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
        """
        s = evoformer_output_dict["single"]
        
        if mask is None:
            # [*, N]
            mask = s.new_ones(s.shape[:-1])

        # [*, N, C_s]
        s = self.layer_norm_s(s)

        # [*, N, N, C_z]
        z = self.layer_norm_z(evoformer_output_dict["pair"])

        z_reference_list = None
        if(_offload_inference):
            refcount = sys.getrefcount(evoformer_output_dict["pair"])
            print(f"[DEBUG] sys.getrefcount(evoformer_output_dict['pair']): {refcount}")
            assert(refcount == 2), f"Expected refcount 2 for evoformer_output_dict['pair'], got {refcount}. This may indicate an unexpected reference or memory leak."
            evoformer_output_dict["pair"] = evoformer_output_dict["pair"].cpu()
            z_reference_list = [z]
            z = None

        # [*, N, C_s]
        s_initial = s
        s = self.linear_in(s)

        # [*, N]
        rigids = Rigid.identity(
            s.shape[:-1], 
            s.dtype, 
            s.device, 
            self.training,
            fmt="quat",
        )
        outputs = []
        for i in range(self.no_blocks):
            # [*, N, C_s]
            s = s + self.ipa(
                s, 
                z, 
                rigids, 
                mask, 
                inplace_safe=inplace_safe,
                _offload_inference=_offload_inference, 
                _z_reference_list=z_reference_list
            )
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)
           
            # [*, N]
            rigids = rigids.compose_q_update_vec(self.bb_update(s))

            # To hew as closely as possible to AlphaFold, we convert our
            # quaternion-based transformations to rotation-matrix ones
            # here
            backb_to_global = Rigid(
                Rotation(
                    rot_mats=rigids.get_rots().get_rot_mats(), 
                    quats=None
                ),
                rigids.get_trans(),
            )

            backb_to_global = backb_to_global.scale_translation(
                self.trans_scale_factor
            )

            # [*, N, 7, 2]
            unnormalized_angles, angles = self.angle_resnet(s, s_initial)

            all_frames_to_global = self.torsion_angles_to_frames(
                backb_to_global,
                angles,
                aatype,
            )

            pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global,
                aatype,
            )

            scaled_rigids = rigids.scale_translation(self.trans_scale_factor)
            
            preds = {
                "frames": scaled_rigids.to_tensor_7(),
                "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": pred_xyz,
                "states": s,
            }

            outputs.append(preds)

            rigids = rigids.stop_rot_gradient()

        del z, z_reference_list
        
        if(_offload_inference):
            evoformer_output_dict["pair"] = (
                evoformer_output_dict["pair"].to(s.device)
            )

        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s

        return outputs

    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    restype_rigid_group_default_frame,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    restype_atom14_to_rigid_group,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    restype_atom14_mask,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
        self, r, f  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )

# define the model
class SAXSMSAAttention(nn.Module):
  def __init__(self,
               c_q: int,
               c_k: int,
               c_v: int,
               c_hidden: int,
               no_heads: int,):
    super(SAXSMSAAttention, self).__init__()
    
    self.c_hidden = c_hidden
    self.no_heads = no_heads

    self.layer_norm_q = LayerNorm(c_q)
    self.layer_norm_k = LayerNorm(c_k)
    
    self.linear_q = Linear(c_q, c_hidden * no_heads)
    self.linear_k = Linear(c_k, c_hidden * no_heads)
    self.linear_v = Linear(c_v, c_hidden * no_heads)
    self.linear_o = Linear(c_hidden * no_heads, c_q)
    self.sigmoid = nn.Sigmoid()  # TODO - do I want to use this for gating?

  def forward(self,
              msa: torch.Tensor, 
              saxs: torch.Tensor, 
              inplace_safe: bool = False,):

    # b is batch size, m is # msa clusters, r is # residues, s is # saxs bins
    q_x = msa.view(msa.shape[0], msa.shape[1]*msa.shape[2], msa.shape[3])  # [b, m*r, c]
    k_x = saxs.unsqueeze(-1)  # [b, s, 1]

    # Normalize inputs
    q_x = self.layer_norm_q(q_x)  # [b, m*r, c]
    kv_x = self.layer_norm_k(k_x)  # [b, s, 1]

    q = self.linear_q(q_x)  # [b, m*r, h*c_hidden]
    k = self.linear_k(kv_x)  # [b, s, h*c_hidden]
    v = self.linear_v(kv_x)  # [b, s, h*c_hidden]

    # reshape for multiple heads
    q = q.view(q.shape[:-1] + (self.no_heads, -1)) # [b, m*r, h, c_hidden]
    k = k.view(k.shape[:-1] + (self.no_heads, -1)) # [b, s, h, c_hidden]
    v = v.view(v.shape[:-1] + (self.no_heads, -1)) # [b, s, h, c_hidden] 

    # transpose heads
    q = q.transpose(1, 2)  # [b, h, m*r, c_hidden]
    k = k.transpose(1, 2)  # [b, h, s, c_hidden]
    v = v.transpose(1, 2)  # [b, h, s, c_hidden]

    # scale query values
    q /= math.sqrt(self.c_hidden)

    # permute last two dims of key for multiplication with query
    k = k.transpose(-2, -1)  # [b, h, c_hidden, s]

    # Compute attention weights
    a = torch.matmul(q, k)  # [b, h, m*r, s]  # NOTE - large matrix - may need to reduce
    a = F.softmax(a, dim=-1)  # [b, h, m*r, s]

    # Compute weighted sum of values
    o = torch.matmul(a, v)  # [b, h, m*r, c_hidden]

    # Flatten final dims
    o = o.transpose(1, 2)  # [b, m*r, h, c_hidden]
    o = o.reshape(o.shape[:-2] + (-1,))  # [b, m*r, h*c_hidden]

    # Transform for output
    o = self.linear_o(o)  # [b, m*r, c]
    o = o.view(msa.shape[:])  # [b, m, r, c]

    return add(msa, o, inplace=inplace_safe)
  
class SAXSPairAttention(nn.Module):
  def __init__(self,
            c_q: int,
            c_k: int,
            c_v: int,
            c_hidden: int,
            no_heads: int,):
    super(SAXSPairAttention, self).__init__()

    self.c_hidden = c_hidden
    self.no_heads = no_heads

    self.layer_norm_q = LayerNorm(c_q)
    self.layer_norm_k = LayerNorm(c_k)
    
    self.linear_q = Linear(c_q, c_hidden * no_heads)
    self.linear_k = Linear(c_k, c_hidden * no_heads)
    self.linear_v = Linear(c_v, c_hidden * no_heads)
    self.linear_o = Linear(c_hidden * no_heads, c_q)
    self.sigmoid = nn.Sigmoid()  # TODO - do I want to use this for gating?

  def forward(self,
            pair: torch.Tensor,
            saxs: torch.Tensor,
            inplace_safe: bool = False,):

    # b is batch size, r is # residues, r is # residues, s is # saxs bins
    q_x = pair.view(pair.shape[0], pair.shape[1]*pair.shape[2], pair.shape[3])  # [b, r*r, c]
    k_x = saxs.unsqueeze(-1)  # [b, s, 1]

    # Normalize inputs
    q_x = self.layer_norm_q(q_x)  # [b, r, r, c]
    kv_x = self.layer_norm_k(k_x)  # [b, s, 1]

    q = self.linear_q(q_x)  # [b, r*r, h*c_hidden]
    k = self.linear_k(kv_x)  # [b, s, h*c_hidden]
    v = self.linear_v(kv_x)  # [b, s, h*c_hidden]

    # reshape for multiple heads
    q = q.view(q.shape[:-1] + (self.no_heads, -1)) # [b, r*r, h, c_hidden]
    k = k.view(k.shape[:-1] + (self.no_heads, -1)) # [b, s, h, c_hidden]
    v = v.view(v.shape[:-1] + (self.no_heads, -1)) # [b, s, h, c_hidden] 

    # transpose heads
    q = q.transpose(1, 2)  # [b, h, r*r, c_hidden]
    k = k.transpose(1, 2)  # [b, h, s, c_hidden]
    v = v.transpose(1, 2)  # [b, h, s, c_hidden]

    # scale query values
    q /= math.sqrt(self.c_hidden)

    # permute last two dims of key for multiplication with query
    k = k.transpose(-2, -1)  # [b, h, c_hidden, s]

    # Compute attention weights
    a = torch.matmul(q, k)  # [b, h, r*r, s]  # NOTE - large matrix - may need to reduce
    a = F.softmax(a, dim=-1)  # [b, h, r*r, s]

    # Compute weighted sum of values
    o = torch.matmul(a, v)  # [b, h, r*r, c_hidden]

    # Flatten final dims
    o = o.transpose(1, 2)  # [b, r*r, h, c_hidden]
    o = o.reshape(o.shape[:-2] + (-1,))  # [b, r*r, h*c_hidden]

    # Transform for output
    o = self.linear_o(o)  # [b, r*r, c]
    o = o.view(pair.shape[:])  # [b, m, r, c]

    return add(pair, o, inplace=inplace_safe)

class SingleOptimizer(nn.Module):
    def __init__(self, c_s: int, c_hidden: int = None):
        """
        Single representation optimizer that applies a linear transformation
        to modify the single representation.
        
        Args:
            c_s: Single representation channel dimension
            c_hidden: Hidden dimension (defaults to c_s if not provided)
        """
        super(SingleOptimizer, self).__init__()
        
        if c_hidden is None:
            c_hidden = c_s
            
        self.c_s = c_s
        self.c_hidden = c_hidden
        
        # Layer normalization for input
        self.layer_norm = LayerNorm(c_s)
        
        # Linear transformation layers
        self.linear_1 = Linear(c_s, c_hidden)
        self.linear_2 = Linear(c_hidden, c_s)
        
        # Activation function
        self.activation = nn.ReLU()
        
    def forward(self, s: torch.Tensor, inplace_safe: bool = False):
        """
        Apply optimization to single representation.
        
        Args:
            s: Single representation tensor [*, N_res, C_s]
            inplace_safe: Whether to use inplace operations
            
        Returns:
            Modified single representation [*, N_res, C_s]
        """
        # Normalize input
        s_norm = self.layer_norm(s)
        
        # Apply linear transformations
        s_hidden = self.activation(self.linear_1(s_norm))
        s_out = self.linear_2(s_hidden)
        
        # Residual connection
        return add(s, s_out, inplace=inplace_safe)

class StructureSAXS(nn.Module):
    """
    Alphafold 2.

    Implements Algorithm 2 (but with training).
    """

    def __init__(self, config):
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)
        """
        super(StructureSAXS, self).__init__()

        self.globals = config.globals
        self.config = config.model
        self.template_config = self.config.template
        self.extra_msa_config = self.config.extra_msa

        # Main trunk + structure module
        self.input_embedder = InputEmbedder(
            **self.config["input_embedder"],
        )
        self.recycling_embedder = RecyclingEmbedder(
            **self.config["recycling_embedder"],
        )
        
        if(self.template_config.enabled):
            self.template_angle_embedder = TemplateAngleEmbedder(
                **self.template_config["template_angle_embedder"],
            )
            self.template_pair_embedder = TemplatePairEmbedder(
                **self.template_config["template_pair_embedder"],
            )
            self.template_pair_stack = TemplatePairStack(
                **self.template_config["template_pair_stack"],
            )
            self.template_pointwise_att = TemplatePointwiseAttention(
                **self.template_config["template_pointwise_attention"],
            )
       
        if(self.extra_msa_config.enabled):
            self.extra_msa_embedder = ExtraMSAEmbedder(
                **self.extra_msa_config["extra_msa_embedder"],
            )
            self.extra_msa_stack = ExtraMSAStack(
                **self.extra_msa_config["extra_msa_stack"],
            )
        
        # saxs attention
        self.saxs_msa_attention = SAXSMSAAttention(
            **self.config["saxs_msa_attention"]
            )
        self.saxs_pair_attention = SAXSPairAttention(
            **self.config["saxs_pair_attention"]
        )
        
        # single representation optimizer
        self.single_optimizer = SingleOptimizer(
            c_s=self.config["evoformer_stack"]["c_s"],
            c_hidden=self.config.get("single_optimizer", {}).get("c_hidden", None)
        )
        
        self.evoformer = EvoformerStack(
            **self.config["evoformer_stack"],
        )
        self.structure_module = StructureModule(
            **self.config["structure_module"],
        )
        self.aux_heads = AuxiliaryHeads(
            self.config["heads"],
        )

    def embed_templates(self, batch, z, pair_mask, templ_dim, inplace_safe): 
        if(self.template_config.offload_templates):
            return embed_templates_offload(self, 
                batch, z, pair_mask, templ_dim, inplace_safe=inplace_safe,
            )
        elif(self.template_config.average_templates):
            return embed_templates_average(self, 
                batch, z, pair_mask, templ_dim, inplace_safe=inplace_safe,
            )

        # Embed the templates one at a time (with a poor man's vmap)
        pair_embeds = []
        n = z.shape[-2]
        n_templ = batch["template_aatype"].shape[templ_dim]

        if(inplace_safe):
            # We'll preallocate the full pair tensor now to avoid manifesting
            # a second copy during the stack later on
            t_pair = z.new_zeros(
                z.shape[:-3] + 
                (n_templ, n, n, self.globals.c_t)
            )

        for i in range(n_templ):
            idx = batch["template_aatype"].new_tensor(i)
            single_template_feats = tensor_tree_map(
                lambda t: torch.index_select(t, templ_dim, idx).squeeze(templ_dim),
                batch,
            )

            # [*, N, N, C_t]
            t = build_template_pair_feat(
                single_template_feats,
                use_unit_vector=self.config.template.use_unit_vector,
                inf=self.config.template.inf,
                eps=self.config.template.eps,
                **self.config.template.distogram,
            ).to(z.dtype)
            t = self.template_pair_embedder(t)

            if(inplace_safe):
                t_pair[..., i, :, :, :] = t
            else:
                pair_embeds.append(t)
            
            del t

        if(not inplace_safe):
            t_pair = torch.stack(pair_embeds, dim=templ_dim)
       
        del pair_embeds

        # [*, S_t, N, N, C_z]
        t = self.template_pair_stack(
            t_pair, 
            pair_mask.unsqueeze(-3).to(dtype=z.dtype), 
            chunk_size=self.globals.chunk_size,
            use_lma=self.globals.use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=self.config._mask_trans,
        )
        del t_pair

        # [*, N, N, C_z]
        t = self.template_pointwise_att(
            t, 
            z, 
            template_mask=batch["template_mask"].to(dtype=z.dtype),
            use_lma=self.globals.use_lma,
        )

        t_mask = torch.sum(batch["template_mask"], dim=-1) > 0
        # Append singletons
        t_mask = t_mask.reshape(
            *t_mask.shape, *([1] * (len(t.shape) - len(t_mask.shape)))
        )

        if(inplace_safe):
            t *= t_mask
        else:
            t = t * t_mask

        ret = {}

        ret.update({"template_pair_embedding": t})

        del t

        if self.config.template.embed_angles:
            template_angle_feat = build_template_angle_feat(
                batch
            )

            # [*, S_t, N, C_m]
            a = self.template_angle_embedder(template_angle_feat)

            ret["template_angle_embedding"] = a

        return ret

    def iteration(self, feats, prevs, _recycle=True):
        # Primary output dictionary
        outputs = {}

        # This needs to be done manually for DeepSpeed's sake
        # dtype = next(self.parameters()).dtype
        # for k in feats:
        #     if(feats[k].dtype == torch.float32):
        #         feats[k] = feats[k].to(dtype=dtype)

        # Grab some data about the input
        batch_dims = feats["target_feat"].shape[:-2]
        no_batch_dims = len(batch_dims)
        n = feats["target_feat"].shape[-2]
        n_seq = feats["msa_feat"].shape[-3]
        
        # Controls whether the model uses in-place operations throughout
        # The dual condition accounts for activation checkpoints
        inplace_safe = not (self.training or torch.is_grad_enabled())

        # Prep some features
        seq_mask = feats["seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        msa_mask = feats["msa_mask"]
        
        ## Initialize the MSA and pair representations

        # m: [*, S_c, N, C_m]
        # z: [*, N, N, C_z]
        m, z = self.input_embedder(
            feats["target_feat"],
            feats["residue_index"],
            feats["msa_feat"],
            inplace_safe=inplace_safe,
        )

        # Unpack the recycling embeddings. Removing them from the list allows 
        # them to be freed further down in this function, saving memory
        m_1_prev, z_prev, x_prev = reversed([prevs.pop() for _ in range(3)])

        # Initialize the recycling embeddings, if needs be 
        if None in [m_1_prev, z_prev, x_prev]:
            # [*, N, C_m]
            m_1_prev = m.new_zeros(
                (*batch_dims, n, self.config.input_embedder.c_m),
                requires_grad=False,
            )

            # [*, N, N, C_z]
            z_prev = z.new_zeros(
                (*batch_dims, n, n, self.config.input_embedder.c_z),
                requires_grad=False,
            )

            # [*, N, 3]
            x_prev = z.new_zeros(
                (*batch_dims, n, residue_constants.atom_type_num, 3),
                requires_grad=False,
            )

        x_prev = pseudo_beta_fn(
            feats["aatype"], x_prev, None
        ).to(dtype=z.dtype)

        # The recycling embedder is memory-intensive, so we offload first
        if(self.globals.offload_inference and inplace_safe):
            m = m.cpu()
            z = z.cpu()

        # m_1_prev_emb: [*, N, C_m]
        # z_prev_emb: [*, N, N, C_z]
        m_1_prev_emb, z_prev_emb = self.recycling_embedder(
            m_1_prev,
            z_prev,
            x_prev,
            inplace_safe=inplace_safe,
        )

        if(self.globals.offload_inference and inplace_safe):
            m = m.to(m_1_prev_emb.device)
            z = z.to(z_prev.device)

        # [*, S_c, N, C_m]
        m[..., 0, :, :] += m_1_prev_emb

        # [*, N, N, C_z]
        z = add(z, z_prev_emb, inplace=inplace_safe)

        # Deletions like these become significant for inference with large N,
        # where they free unused tensors and remove references to others such
        # that they can be offloaded later
        del m_1_prev, z_prev, x_prev, m_1_prev_emb, z_prev_emb

        # Embed the templates + merge with MSA/pair embeddings
        if self.config.template.enabled: 
            template_feats = {
                k: v for k, v in feats.items() if k.startswith("template_")
            }
            template_embeds = self.embed_templates(
                template_feats,
                z,
                pair_mask.to(dtype=z.dtype),
                no_batch_dims,
                inplace_safe=inplace_safe,
            )

            # [*, N, N, C_z]
            z = add(z,
                template_embeds.pop("template_pair_embedding"),
                inplace_safe,
            )

            if "template_angle_embedding" in template_embeds:
                # [*, S = S_c + S_t, N, C_m]
                m = torch.cat(
                    [m, template_embeds["template_angle_embedding"]], 
                    dim=-3
                )

                # [*, S, N]
                torsion_angles_mask = feats["template_torsion_angles_mask"]
                msa_mask = torch.cat(
                    [feats["msa_mask"], torsion_angles_mask[..., 2]], 
                    dim=-2
                )

        # Embed extra MSA features + merge with pairwise embeddings
        if self.config.extra_msa.enabled:
            # [*, S_e, N, C_e]
            a = self.extra_msa_embedder(build_extra_msa_feat(feats))

            if(self.globals.offload_inference):
                # To allow the extra MSA stack (and later the evoformer) to
                # offload its inputs, we remove all references to them here
                input_tensors = [a, z]
                del a, z
    
                # [*, N, N, C_z]
                z = self.extra_msa_stack._forward_offload(
                    input_tensors,
                    msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
                    chunk_size=self.globals.chunk_size,
                    use_lma=self.globals.use_lma,
                    pair_mask=pair_mask.to(dtype=m.dtype),
                    _mask_trans=self.config._mask_trans,
                )
    
                del input_tensors
            else:
                # [*, N, N, C_z]
                z = self.extra_msa_stack(
                    a, z,
                    msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
                    chunk_size=self.globals.chunk_size,
                    use_lma=self.globals.use_lma,
                    pair_mask=pair_mask.to(dtype=m.dtype),
                    inplace_safe=inplace_safe,
                    _mask_trans=self.config._mask_trans,
                )

        # Run SAXS attention module to get modified MSA embedding
        m = self.saxs_msa_attention(msa=m, 
                                    saxs=feats['saxs'], 
                                    inplace_safe=inplace_safe)
        
        z = self.saxs_pair_attention(pair=z,
                                     saxs=feats['saxs'],
                                     inplace_safe=inplace_safe)

        # Run MSA + pair embeddings through the trunk of the network
        # m: [*, S, N, C_m]
        # z: [*, N, N, C_z]
        # s: [*, N, C_s]          
        if(self.globals.offload_inference):
            input_tensors = [m, z]
            del m, z
            m, z, s = self.evoformer._forward_offload(
                input_tensors,
                msa_mask=msa_mask.to(dtype=input_tensors[0].dtype),
                pair_mask=pair_mask.to(dtype=input_tensors[1].dtype),
                chunk_size=self.globals.chunk_size,
                use_lma=self.globals.use_lma,
                _mask_trans=self.config._mask_trans,
            )
    
            del input_tensors
        else:
            m, z, s = self.evoformer(
                m,
                z,
                msa_mask=msa_mask.to(dtype=m.dtype),
                pair_mask=pair_mask.to(dtype=z.dtype),
                chunk_size=self.globals.chunk_size,
                use_lma=self.globals.use_lma,
                use_flash=self.globals.use_flash,
                inplace_safe=inplace_safe,
                _mask_trans=self.config._mask_trans,
            )

        outputs["msa"] = m[..., :n_seq, :, :]
        outputs["pair"] = z
        outputs["single"] = s

        # Apply single representation optimization
        s = self.single_optimizer(s, inplace_safe=inplace_safe)
        outputs["single"] = s

        del z

        # Predict 3D structure
        outputs["sm"] = self.structure_module(
            outputs,
            feats["aatype"],
            mask=feats["seq_mask"].to(dtype=s.dtype),
            inplace_safe=inplace_safe,
            _offload_inference=self.globals.offload_inference,
        )
        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"][-1], feats
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

        # Save embeddings for use during the next recycling iteration

        # [*, N, C_m]
        m_1_prev = m[..., 0, :, :]

        # [*, N, N, C_z]
        z_prev = outputs["pair"]

        # [*, N, 3]
        x_prev = outputs["final_atom_positions"]

        return outputs, m_1_prev, z_prev, x_prev

    def forward(self, batch):
        """
        Args:
            batch:
                Dictionary of arguments outlined in Algorithm 2. Keys must
                include the official names of the features in the
                supplement subsection 1.2.9.

                The final dimension of each input must have length equal to
                the number of recycling iterations.

                Features (without the recycling dimension):

                    "aatype" ([*, N_res]):
                        Contrary to the supplement, this tensor of residue
                        indices is not one-hot.
                    "target_feat" ([*, N_res, C_tf])
                        One-hot encoding of the target sequence. C_tf is
                        config.model.input_embedder.tf_dim.
                    "residue_index" ([*, N_res])
                        Tensor whose final dimension consists of
                        consecutive indices from 0 to N_res.
                    "msa_feat" ([*, N_seq, N_res, C_msa])
                        MSA features, constructed as in the supplement.
                        C_msa is config.model.input_embedder.msa_dim.
                    "seq_mask" ([*, N_res])
                        1-D sequence mask
                    "msa_mask" ([*, N_seq, N_res])
                        MSA mask
                    "pair_mask" ([*, N_res, N_res])
                        2-D pair mask
                    "extra_msa_mask" ([*, N_extra, N_res])
                        Extra MSA mask
                    "template_mask" ([*, N_templ])
                        Template mask (on the level of templates, not
                        residues)
                    "template_aatype" ([*, N_templ, N_res])
                        Tensor of template residue indices (indices greater
                        than 19 are clamped to 20 (Unknown))
                    "template_all_atom_positions"
                        ([*, N_templ, N_res, 37, 3])
                        Template atom coordinates in atom37 format
                    "template_all_atom_mask" ([*, N_templ, N_res, 37])
                        Template atom coordinate mask
                    "template_pseudo_beta" ([*, N_templ, N_res, 3])
                        Positions of template carbon "pseudo-beta" atoms
                        (i.e. C_beta for all residues but glycine, for
                        for which C_alpha is used instead)
                    "template_pseudo_beta_mask" ([*, N_templ, N_res])
                        Pseudo-beta mask
        """
        # Initialize recycling embeddings
        m_1_prev, z_prev, x_prev = None, None, None
        prevs = [m_1_prev, z_prev, x_prev]

        is_grad_enabled = torch.is_grad_enabled()

        # Main recycling loop
        num_iters = batch["aatype"].shape[-1]
        for cycle_no in range(num_iters): 
            # Select the features for the current recycling cycle
            def fetch_cur_batch(t):
              return t[..., cycle_no]
            feats = tensor_tree_map(fetch_cur_batch, batch)

            # Enable grad iff we're training and it's the final recycling layer
            is_final_iter = cycle_no == (num_iters - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                if is_final_iter:
                    # Sidestep AMP bug (PyTorch issue #65766)
                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()

                # Run the next iteration of the model
                outputs, m_1_prev, z_prev, x_prev = self.iteration(
                    feats,
                    prevs,
                    _recycle=(num_iters > 1)
                )

                if(not is_final_iter):
                    del outputs
                    prevs = [m_1_prev, z_prev, x_prev]
                    del m_1_prev, z_prev, x_prev

        # Run auxiliary heads
        outputs.update(self.aux_heads(outputs))

        return outputs