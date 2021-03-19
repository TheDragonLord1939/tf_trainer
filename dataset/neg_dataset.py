from dataset.rec_dataset import RecDataset


class NegDataset(RecDataset):

    def _de_prebatch_parser(self, samples):
        features, labels = super()._de_prebatch_parser(samples)
        return features, -labels + 1
