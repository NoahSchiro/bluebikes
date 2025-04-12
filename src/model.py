from torch import nn
import torch

class BlueBikesModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # Predict arrivals + departures
        self.linear = nn.Linear(hidden_size, 2)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.linear(hidden[-1])  # Use last hidden state for prediction

if __name__=="__main__":

    model = BlueBikesModel(
        input_size=6,
        hidden_size=32,
        num_layers=2
    )

    data = torch.randn((2, 24, 6))
    output = model.forward(data)

    print(output)
