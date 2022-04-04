"""MAESTRO dataset loader"""

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{
  hawthorne2018enabling,
  title={Enabling Factorized Piano Music Modeling and Generation with the {MAESTRO} Dataset},
  author={Curtis Hawthorne and Andriy Stasyuk and Adam Roberts and Ian Simon and Cheng-Zhi Anna Huang and Sander Dieleman and Erich Elsen and Jesse Engel and Douglas Eck},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=r1lYRjC9F7},
}
"""


_DESCRIPTION = """\
The Google MAESTRO piano music dataset.
"""


class MaestroConfig(datasets.BuilderConfig):
    """BuilderConfig for MAESTRO."""

    def __init__(self, **kwargs):
        """BuilderConfig for MAESTRO.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MaestroConfig, self).__init__(**kwargs)


class Maestro(datasets.GeneratorBasedBuilder):
    """MAESTRO piano music dataset."""

    BUILDER_CONFIGS = [
        MaestroConfig(
            name="maestro-v1.0.0",
            version=datasets.Version("1.0.0", ""),
            description="Tokenized MAESTRO",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://magenta.tensorflow.org/datasets/maestro",
            citation=_CITATION
        )

    def _split_generators(self, dl_manager):
        print(self.config)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"data_file": f"{self.config.data_dir}/maestro-train.txt"}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"data_file": f"{self.config.data_dir}/maestro-valid.txt"}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"data_file": f"{self.config.data_dir}/maestro-test.txt"}),
        ]

    def _generate_examples(self, data_file):
        """This function returns the MAESTRO text in the discretized, tokenized form."""
        logger.info("generating examples from = %s", data_file)
        with open(data_file, encoding="utf-8") as f:
            for idx, row in enumerate(f):
                yield idx, {"text": row}

