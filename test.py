import torch
import unittest
from train import build_model

class TestModel(unittest.TestCase):
    def test_model_output_shape(self):
        # Verify that the model outputs a tensor of shape [batch_size, num_classes]
        num_classes = 10
        model = build_model(num_classes=num_classes)
        model.eval()
        # Create a dummy input (batch size=1, 3 channels, 224x224)
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, num_classes), "Output shape is incorrect.")

if __name__ == '__main__':
    unittest.main()
