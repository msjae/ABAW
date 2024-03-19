import torch.nn as nn
from transformers import ViTImageProcessor, ViTModel, ViTForImageClassification
from torchvision import models
import torch

class linear_model(nn.Module):
    def __init__(self, cfg):
        super(linear_model, self).__init__()
        
        # 모델 아키텍처를 구성하는 레이어들을 초기화
        models = [nn.Flatten()]
        if cfg['return_vis'] and cfg['return_aud']:
            models.append(nn.Linear(cfg['seq_size'] * cfg['embedding_size'] * 2, cfg['num_classes']))
        else:
            models.append(nn.Linear(cfg['seq_size'] * cfg['embedding_size'], cfg['num_classes']))        
        # VA(Valence-Arousal) 작업에 대한 설정
        if cfg['task'] == 'VA':
            models.append(nn.Tanh())
        elif cfg['task'] == 'EXPR':
            models.append(nn.Softmax(dim=1))
        else:
            models.append(nn.Sigmoid())
        # 정의된 레이어들을 Sequential 모듈로 결합
        self.model = nn.Sequential(*models)

    def forward(self, x):
        # 입력 x를 모델에 통과시켜 결과를 반환
        return self.model(x)

class lstm_model(nn.Module):
    def __init__(self, cfg):
        super(lstm_model, self).__init__()
        
        # 모델 아키텍처를 구성하는 레이어들을 초기화
        input_size = cfg['embedding_size']
        self.lstm = nn.LSTM(input_size, 1024, 1, batch_first=True, dropout=cfg['dropout'])
        self.fc = nn.Linear(1024, cfg['num_classes'])
        # VA(Valence-Arousal) 작업에 대한 설정
        if cfg['task'] == 'VA':
            self.tanh = nn.Tanh()
        self.cfg = cfg
    def forward(self, x):
        # 입력 x를 모델에 통과시켜 결과를 반환
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        if self.cfg['task'] == 'VA':
            x = self.tanh(x)
        return x
    
class ImageFeatureTransformer(nn.Module):
    def __init__(self, vit_model_name, cfg, device='cuda:1'):
        super(ImageFeatureTransformer, self).__init__()
        self.device = torch.device(device)
        self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTModel.from_pretrained(vit_model_name).to(self.device)
        self.model.eval()
        # # trainable False
        # for param in self.model.parameters():
        #     param.requires_grad = False
        self.head = linear_model(cfg).to(self.device)

    def forward(self, images):
        inputs = self.image_processor(images=images, return_tensors="pt", do_rescale=False, do_resize=False).to(self.device)
        outputs = self.model(**inputs)
        features = outputs.pooler_output
        output = self.head(features)
        return output

class ImageTransformer(nn.Module):
    def __init__(self, vit_model_name, cfg, device='cuda:1'):
        super(ImageTransformer, self).__init__()
        self.device = torch.device(device)
        self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTForImageClassification.from_pretrained(vit_model_name).to(self.device)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, cfg['num_classes']).to(self.device)
    def forward(self, images):
        inputs = self.image_processor(images=images, return_tensors="pt", do_rescale=False, do_resize=False).to(self.device)
        outputs = self.model(**inputs).logits
        return outputs

class resnet_model(nn.Module):
    def __init__(self, cfg):
        super(resnet_model, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, cfg['num_classes'])
        self.cfg = cfg
    def forward(self, x):
        x = self.model(x)
        if self.cfg['task'] == 'VA':
            x = torch.tanh(x)
        elif self.cfg['task'] == 'EXPR':
            x = torch.softmax(x, dim=1)
        else:
            x = torch.sigmoid(x)
        return x