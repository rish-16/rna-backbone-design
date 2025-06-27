import numpy as np
import torch, os

from rna_backbone_design.tools.rhofold_api.rhofold.rf import RhoFold
from rna_backbone_design.tools.rhofold_api.rhofold.config import rhofold_config # NOTE: can remove this
from rna_backbone_design.tools.rhofold_api.rhofold.utils.alphabet import get_features

CKPT = "./rna_backbone_design/tools/rhofold_api/checkpoints/RhoFold_pretrained.pt"

with torch.no_grad():
    model = RhoFold(rhofold_config)
    model.load_state_dict(torch.load(CKPT, map_location=torch.device('cpu'))['model'])
    model.eval()

    input_a3m = input_fas = "./sample3.fasta"

    device = "cuda:1"
    model = model.to(device)
    data_dict = get_features(input_fas, input_a3m)

    outputs = model(tokens=data_dict['tokens'].to(device), rna_fm_tokens=data_dict['rna_fm_tokens'].to(device), seq=data_dict['seq'])
    
    output = outputs[-1]
    output_dir = "rna_backbone_design/tools/rhofold_api/sample3_pdb_pred"
    os.makedirs(output_dir, exist_ok=True)
    unrelaxed_model = os.path.join(output_dir, 'unrelaxed_sample3_model.pdb')
    node_cords_pred = output['cord_tns_pred'][-1].squeeze(0)

    model.structure_module.converter.export_pdb_file(
                                        data_dict['seq'],
                                        node_cords_pred.data.cpu().numpy(),
                                        path=unrelaxed_model, 
                                        chain_id=None,
                                        confidence=output['plddt'][0].data.cpu().numpy(),
                                        logger=None
                                    )