import random

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

import sys
sys.path.append("/home/tianyi/lm-evaluation-harness")
from models.tnlg import PhiForCausalLM, LlamaForCausalLM, Prompter

torch.set_default_dtype(torch.float16)

class TNLGLM(BaseLM):
    def __init__(self, ckpt_path="/home/mohamed/Codes/TNLG_End2End/checkpoints/tnlg_v4_7b_fp16.pt", is_pruned=True, _device: str = 'cuda'):
        super().__init__()
        self.out_seq_len = 300
        self.max_seq_len = 4096
        self.batch_size_per_gpu = 4
        self._device = _device
        self.tokenizer = TNLGTokenizer()
        self.prompter = Prompter()

        if is_pruned:
            # self.model = torch.load(ckpt_path, map_location=_device)['model']
            self.model = torch.load(ckpt_path, map_location=_device)
        else:
            n_layers, n_heads, hidden_size, cur_seq_len, batch_size, max_seq_len, output_len = 32, 32, 4096, 128, 2, 4096, 100352
            self.model = TNLG(n_layers=n_layers,
                    vocab_size=output_len,
                    hidden_size=hidden_size,
                    n_heads=n_heads,
                    seq_len=max_seq_len,
                    device=_device).to(_device)
            self.model.load_state_dict(torch.load(ckpt_path, map_location=_device))
        self.model.eval()

    @property
    def max_gen_toks(self):
        return self.out_seq_len
    
    @property
    def device(self):
        return self._device
    
    @property
    def eot_token_id(self):
        return self.tokenizer.eos_id
    
    @property
    def max_length(self):
        return self.max_seq_len
    
    @property
    def batch_size(self):
        return self.batch_size_per_gpu
    
    def tok_encode(self, string: str):
        prompt = self.prompter.generate_prompt(string)
        tokens = self.tokenizer.tokenize(prompt)

        torch.cuda.empty_cache()
        return tokens
    
    def tok_decode(self, tokens):
        return self.tokenizer.detokenize(tokens)
    
    def _model_call(self, inps):
        with torch.no_grad():
            logits, _, _ = self.model(inps)
            torch.cuda.empty_cache()
            return logits
        
    def _model_generate(self, context, max_length, eos_token_id):
        import pdb; pdb.set_trace()
        pass
