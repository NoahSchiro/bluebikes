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

    # Takes in data and stats about the data and
    # performs preprocessing conducive to ML
    def preprocess(self, data, stats):

        # For month, day, and hour, we will use a cyclical encoding
        def cyclical(data, max):
            return torch.sin((2 * torch.pi * ((data-1) / max)))

        # Month
        data[:, :, 1] = cyclical(data[:, :, 1], 12.0)
        # Day
        data[:, :, 2] = cyclical(data[:, :, 2], 31.0)
        # Hour 
        data[:, :, 3] = cyclical(data[:, :, 3], 24.0)

        # Normalize year based on min and max (linear transformation to [0,1])
        year_min = stats["year"]["min"]
        year_max = stats["year"]["max"]
        year_range = year_max - year_min
        data[:, :, 0] = (data[:, :, 0] - year_min) / year_range

        return data

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.linear(hidden[-1])  # Use last hidden state for prediction

    def inference(self, data, stats):
        data = self.preprocess(data, stats)

        prediction = self.forward(data).round().relu()

        # Data comes back in shape [batch_size, 2]
        # [[arrival, departures],
        #  ...
        #  [arrivals, departures]]
        return prediction.tolist()


if __name__=="__main__":

    model = BlueBikesModel(
        input_size=7,
        hidden_size=32,
        num_layers=2
    )

    year_norm = {
        "year": {
            "min" : 2010,
            "max" : 2025
        },
    }

    data = torch.randn((2, 24, 7))
    output = model.inference(data, year_norm)

    print(output)
