def foveate(self, x, l):
		"""
		Extract `k` square patches of size `g`, centered
		at location `l`. The initial patch is a square of
		size `g`, and each subsequent patch is a square
		whose side is `s` times the size of the previous
		patch.

		The `k` patches are finally resized to (g, g) and
		concatenated into a tensor of shape (B, k, g, g, C).
		"""
		phi = []
		size = self.g

		# extract k patches of increasing size
		for i in range(self.k):
			phi.append(self.extract_patch(x, l, size,i)) ## send i as well

		# resize the patches to squares of size g
		for i in range(1, len(phi)):
			k = phi[i].shape[-1] // self.g
			phi[i] = F.avg_pool2d(phi[i], k)

		# concatenate into a single tensor and flatten
		phi = torch.cat(phi, 1)
		phi = phi.view(phi.shape[0], -1)

		return phi

	def extract_patch(self, x, l, size, j): ## 
		"""
		Extract a single patch for each image in the
		minibatch `x`.

		Args
		----
		- x: a 4D Tensor of shape (B, H, W, C). The minibatch
		  of images.
		- l: a 2D Tensor of shape (B, 2).
		- size: a scalar defining the size of the extracted patch.

		Returns
		-------
		- patch: a 4D Tensor of shape (B, size, size, C)
		"""
		B, C, H, W = x.shape

		# denormalize coords of patch center
		coords = self.denormalize(H, l)

		# compute top left corner of patch
		patch_x = coords[:, 0] - (size // 2)
		patch_y = coords[:, 1] - (size // 2)

		# loop through mini-batch and extract
		patch = []
		if(j==0):
			for i in range(B):
				im = x[i].unsqueeze(dim=0)
				T = im.shape[-1]

				# compute slice indices
				from_x, to_x = patch_x[i], patch_x[i] + size
				from_y, to_y = patch_y[i], patch_y[i] + size

				# cast to ints
				from_x, to_x = from_x.data.item(), to_x.data.item()
				from_y, to_y = from_y.data.item(), to_y.data.item()

				# pad tensor in case exceeds
				if self.exceeds(from_x, to_x, from_y, to_y, T):
					pad_dims = (
						size//2+1, size//2+1,
						size//2+1, size//2+1,
						0, 0,
						0, 0,
					)
					im = F.pad(im, pad_dims, "constant", 0)

					# add correction factor
					from_x += (size//2+1)
					to_x += (size//2+1)
					from_y += (size//2+1)
					to_y += (size//2+1)

				# and finally extract
				patch.append(im[:, :, from_y:to_y, from_x:to_x])
		else:
			for i in range(B):
				im = x[i].unsqueeze(dim=0)
				T = im.shape[-1]

				# compute slice indices
				from_x, to_x = patch_x[i], patch_x[i] + size
				from_y, to_y = patch_y[i], patch_y[i] + size

				# cast to ints
				from_x, to_x = from_x.data.item(), to_x.data.item()
				from_y, to_y = from_y.data.item(), to_y.data.item()

				# pad tensor in case exceeds
				if self.exceeds(from_x, to_x, from_y, to_y, T):
					pad_dims = (
						size//2+1, size//2+1,
						size//2+1, size//2+1,
						0, 0,
						0, 0,
					)
					im = F.pad(im, pad_dims, "constant", 0)

					# add correction factor
					from_x += (size//2+1)
					to_x += (size//2+1)
					from_y += (size//2+1)
					to_y += (size//2+1)

				# and finally extract
				patch.append(im[:, :, from_y:from_y+size//4+1, from_x:to_x]) ## appending top strip
				patch.append(im[:, :, 3*from_y//4:to_y, from_x:to_x]) ## appending bottom strip
				patch.append(im[:, :,from_y+size//4+1:3*from_y//4, from_x:3*from_x//4+1]) ## appending left strip
				patch.append(im[:, :,from_y+size//4+1:3*from_y//4, 3*from_x//4+1:to_x]) ## appending right strip

		# concatenate into a single tensor
		patch = torch.cat(patch)

		return patch