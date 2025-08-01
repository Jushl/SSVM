class MultithreadBatchParallelSampler:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.data_info = []
        global_offset = 0
        for data in dataset.data_structure:
            self.data_info.append({
                'start_idx': global_offset,
                'end_idx': global_offset + data['num'],
                'num': data['num']
            })
            global_offset += data['num']

        self.num_data = len(self.data_info)

    def __iter__(self):
        for group_start in range(0, self.num_data, self.batch_size):
            group_end = min(group_start + self.batch_size, self.num_data)
            group_data = self.data_info[group_start:group_end]
            min_images = min(f['num'] for f in group_data)
            for img_pos in range(min_images):
                batch_indices = []
                for data in group_data:
                    absolute_idx = data['start_idx'] + img_pos
                    batch_indices.append(absolute_idx)

                yield batch_indices

    def __len__(self):
        total_batches = 0
        for group_start in range(0, self.num_data, self.batch_size):
            group_end = min(group_start + self.batch_size, self.num_data)
            min_images = min(f['num'] for f in self.data_info[group_start:group_end])
            total_batches += min_images
        return total_batches