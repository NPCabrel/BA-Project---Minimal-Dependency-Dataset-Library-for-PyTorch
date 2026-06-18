import torch
import random
import time

class DummyModel:
	"""
	Simiulates GPU compute delay for data loading benchmarks.
	Uses calibration parameters (mean_ms, std_ms) with optional Gaussian noise.
	Args: 
		mean_ms: Mean compute delay in milliseconds (from calibration)
		std_ms: Standard deviation of compute delay in milliseconds
		input_size: Expected input image size (default 224)
		num_classes: Number of output classes (default 1000 for ImageNet)
	 """


	def __init__(self, mean_ms, std_ms=0.0, input_size=224, num_classes=1000):
		# Converting ms to seconds for timesleep()
		self.delay = mean_ms / 1000.0
		self.std = std_ms / 1000.0
		self.input_size = input_size
		self.num_classes = num_classes

	def __call__(self, batch):
		"""
		Simulate forward pass.
		batch: tensor of shape (batch_size, 3, input_size, input_size)
		Returns: fake logits of shape (batch_size, num_classes)"""

		# Calculate delay (variable if std > 0, constant if std = 0)

		if self.std > 0:
			current_delay = max(0.0, random.gauss(self.delay, self.std))
		else:
			current_delay = self.delay

		# Simulate GPU compute
		time.sleep(current_delay)

		#return fake outputs (like a real model would)

		return torch.randn(batch.size(0), self.num_classes)




#----QUICK TEST
if __name__ == "__main__":
	print("Testing DummyModel...")
	model = DummyModel(mean_ms=598.49, std_ms=0.37)
	fake_batch = torch.randn(32, 3, 224, 224)

	start = time.time()
	output = model(fake_batch)
	elapsed = time.time() - start


	print(f"Input Shape: {fake_batch.shape}")
	print(f"Output shape: {output.shape}")
	print(f"Delay: {elapsed*1000:.1f} ms (expected ~598 ms)")
	print("DummyModel works!")
