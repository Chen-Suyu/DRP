class Qwen2PromptTemplate:
    def __init__(self, system_prompt=None):
        self.system_prompt = system_prompt

    def build_prompt(self, user_message, system_prompt=None):
        sys_prompt = system_prompt if system_prompt is not None else self.system_prompt
        if sys_prompt is not None:
            SYS = f"<|im_start|>system\n{sys_prompt}<|im_end|>"
        else:
            SYS = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"

        CONVO = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        return SYS + CONVO


class DeepSeekR1PromptTemplate:
    def __init__(self):
        pass

    def build_prompt(self, user_message):
        return f"bos_token<|User|>{user_message}<|Assistant|><think>\n"
