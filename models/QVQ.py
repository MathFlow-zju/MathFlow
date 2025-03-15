from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image

class QVQModel:
    def __init__(self, session_len=10086, tensor_parallel=4):
        """
        初始化 InternVL Turbomind 推理引擎
        
        Args:
            model_path (str): 模型路径
            session_len (int): 会话最大长度
            tensor_parallel (int): 张量并行度
        """
        self.model_path = f"/home/ecs-user/nas_original_data/csh/MODEL/Qwen/QVQ-72B-Preview"
        # 配置生成参数
        self.generation_config = GenerationConfig(
            max_new_tokens=1024,
            do_sample=True,  # 确保设置为 True 以启用采样
            temperature=0.7,
            top_p=0.8,
            top_k=40,
            repetition_penalty=1.1
        )

        self.pipe = pipeline(
            self.model_path,
            backend_config=TurbomindEngineConfig(
                session_len=session_len,
                tp=tensor_parallel,
                max_prefill_token_num=10086,
                num_tokens_per_iter=1024,
                max_new_tokens = 1024
            ),
            generation_config=self.generation_config
        )
        # self.pipe.generation_config = GenerationConfig(
        #     max_new_tokens=2048,
        #     do_sample=True,  # 确保设置为 True 以启用采样
        #     temperature=0.7,
        #     top_p=0.8,
        #     top_k=40,
        #     repetition_penalty=1.1
        # )
        # print(self.pipe.generation_config)

    def get_response(self, image_path, prompt):
        """
        获取单个问题的回答
        
        Args:
            prompt (str): 问题文本
            image_path (str): 图片路径
            
        Returns:
            str: 模型的回答
        """
        gen_config = GenerationConfig(top_k=40, top_p=0.8, temperature=0.6, max_new_tokens=4096)
        if image_path:
            image = load_image(image_path)
            return self.pipe((prompt, image), gen_config=gen_config).text
        else:
            return self.pipe(prompt, gen_config=gen_config).text
        

