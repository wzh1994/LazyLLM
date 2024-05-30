import os
import json
import random

import lazyllm
from lazyllm import launchers, LazyLLMCMD, ArgsDict, LOG
from .base import LazyLLMDeployBase, verify_fastapi_func


class Vllm(LazyLLMDeployBase):
    input_key_name = 'prompt'
    default_headers = {'Content-Type': 'application/json'}
    message_format = {
        input_key_name: 'Who are you ?',
        'stream': False,
        'stop': ['<|im_end|>', '<|im_start|>', '</s>', '<|assistant|>', '<|user|>', '<|system|>', '<eos>'],
        'skip_special_tokens': False,
        'temperature': 0.01,
        'top_p': 0.8,
        'max_tokens': 1024
    }

    def __init__(self,
                 trust_remote_code=True,
                 launcher=launchers.remote,
                 stream=False,
                 **kw,
                 ):
        super().__init__(launcher=launcher)
        self.kw = ArgsDict({
            'max-model-len': 32968,
            'dtype': 'auto',
            'kv-cache-dtype': 'auto',
            'tokenizer-mode': 'auto',
            'device': 'auto',
            'block-size': 16,
            'tensor-parallel-size': 1,
            'seed': 0,
            'port': 'auto',
            'host': '0.0.0.0',
        })
        self.trust_remote_code = trust_remote_code
        self.kw.check_and_update(kw)

    def cmd(self, model_dir=None, base_model=None):
        if not os.path.exists(model_dir) or \
            not any(filename.endswith('.bin') or filename.endswith('.safetensors')
                    for filename in os.listdir(model_dir)):
            if not model_dir:
                LOG.warning(f"Note! That model_dir({model_dir}) is an invalid path, "
                            f"base_model({base_model}) will be used")
            model_dir = base_model

        def impl():
            if not self.kw['port'] or self.kw['port'] == 'auto':
                self.kw['port'] = random.randint(30000, 40000)

            cmd = f'python -m vllm.entrypoints.api_server --model {model_dir} '
            cmd += self.kw.parse_kwargs()
            if self.trust_remote_code:
                cmd += ' --trust-remote-code '
            return cmd

        return LazyLLMCMD(cmd=impl, return_value=self.geturl, checkf=verify_fastapi_func)

    def geturl(self, job=None):
        if job is None:
            job = self.job
        if lazyllm.config['mode'] == lazyllm.Mode.Display:
            return 'http://{ip}:{port}/generate'
        else:
            return f'http://{job.get_jobip()}:{self.kw["port"]}/generate'

    @staticmethod
    def extract_result(x):
        return json.loads(x)['text'][0]
