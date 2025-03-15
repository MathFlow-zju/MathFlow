from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

class InternVL2Inference:
    def __init__(self, model_ID, session_len=10086, tensor_parallel=8):
        """
        初始化 InternVL Turbomind 推理引擎
        
        Args:
            model_path (str): 模型路径
            session_len (int): 会话最大长度
            tensor_parallel (int): 张量并行度
        """
        self.model_path = f"/home/ecs-user/nas_original_data/csh/MODEL/OpenGVLab/{model_ID}"
        self.pipe = pipeline(
            self.model_path,
            backend_config=TurbomindEngineConfig(
                session_len=session_len,
                tp=tensor_parallel
            )
        )
    
    def get_response(self, image_path, prompt):
        """
        获取单个问题的回答
        
        Args:
            prompt (str): 问题文本
            image_path (str): 图片路径
            
        Returns:
            str: 模型的回答
        """
        if image_path:
            image = load_image(image_path)
            return self.pipe((prompt, image)).text
        else:
            return self.pipe(prompt).text
        


