import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridCNNDQN(nn.Module):
    """
    6G 하이브리드 핸드오버를 위한 시각적 공간 인지 기반 강화학습 신경망
    논문 4.2절(Spatial Awareness Module) 및 4.3절(Decision Making Module) 구현체
    """

    def __init__(self, num_actions=7, vector_dim=5):
        super(HybridCNNDQN, self).__init__()

        # --- [1] 공간 인지 모듈 (Spatial Awareness Module, CNN) ---
        # 입력: (Batch, 3, 100, 100) 형태의 3채널 2D 공간 맵

        self.conv_layers = nn.Sequential(
            # Conv Layer 1: 큰 특징(건물 덩어리 등)을 빠르게 추출
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 출력: (16, 25, 25)

            # Conv Layer 2: 세밀한 특징(위성 궤적, 단말 궤적 꼬리 등) 추출
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # 출력: (32, 6, 6)
        )

        # 특징 맵 평탄화(Flatten) 시 생성되는 1차원 벡터의 크기 계산
        self.cnn_out_dim = 32 * 6 * 6  # 1152 차원의 잠재 표현(Latent Representation)

        # --- [2] 의사결정 모듈 (Decision Making Module, MLP) ---
        # CNN이 추출한 공간 특징 벡터(1152) + 실시간 네트워크 상태 벡터(vector_dim=5) 병합
        self.fc_input_dim = self.cnn_out_dim + vector_dim

        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            # 최종 출력: 6개의 타겟 노드(LEO 1개, GBS 5개)에 대한 Q-value
            nn.Linear(128, num_actions)
        )

    def forward(self, image_input, vector_input):
        """
        순전파(Forward Pass) 연산
        :param image_input: [Batch_size, 3, 100, 100] 공간 맵 텐서
        :param vector_input: [Batch_size, 5] 네트워크 수치 벡터 (SINR, 속도 등)
        :return: [Batch_size, 6] 각 행동에 대한 Q-value
        """
        # 1. 2D 공간 맵을 CNN에 통과시켜 특징 추출 (논문의 h_t^{spatial})
        cnn_features = self.conv_layers(image_input)

        # 2. 특징 맵을 1차원 벡터로 평탄화
        cnn_features = torch.flatten(cnn_features, start_dim=1)

        # 3. 공간 특징 벡터와 네트워크 수치 벡터 병합 (논문의 z_t = h_t^{spatial} ⊕ v_t)
        # dim=1을 기준으로 옆으로 이어 붙임 (Concatenation)
        fused_state = torch.cat((cnn_features, vector_input), dim=1)

        # 4. 병합된 상태 벡터를 평가 네트워크(MLP)에 통과시켜 Q-value 도출
        q_values = self.fc_layers(fused_state)

        return q_values


# --- 모델 생성 및 파라미터 수 확인용 간단한 테스트 스크립트 ---
if __name__ == "__main__":
    # 임의의 더미 데이터 생성 (배치 크기=1 가정)
    dummy_image = torch.rand((1, 3, 100, 100))
    dummy_vector = torch.rand((1, 5))

    # 모델 인스턴스화
    model = HybridCNNDQN(num_actions=6, vector_dim=5)

    # Q-value 추론 테스트
    q_vals = model(dummy_image, dummy_vector)

    print(f"네트워크 구조:\n{model}")
    print(f"입력 이미지 크기: {dummy_image.shape}")
    print(f"입력 벡터 크기: {dummy_vector.shape}")
    print(f"출력 Q-value 크기: {q_vals.shape}")

    # 총 파라미터 수 계산 (논문 6.4절에서 "약 18만 개"라고 주장한 부분의 근거)
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"학습 가능한 총 파라미터 수: {total_params:,} 개")
