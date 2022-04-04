"""FMA dataset loader"""

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
    @inproceedings{defferrard2016fma,
      author    = {Micha{\"{e}}l Defferrard and
                   Kirell Benzi and
                   Pierre Vandergheynst and
                   Xavier Bresson},
      title     = {{FMA:} {A} Dataset for Music Analysis},
      booktitle = {ISMIR},
      year      = 2017
    }
"""


_DESCRIPTION = """\
The FMA music dataset.
"""


class FMAConfig(datasets.BuilderConfig):
    """BuilderConfig for FMA."""

    def __init__(self, **kwargs):
        """BuilderConfig for FMA.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(FMAConfig, self).__init__(**kwargs)


class FMA(datasets.GeneratorBasedBuilder):
    """FMA music dataset."""

    BUILDER_CONFIGS = [
        FMAConfig(
            name="fma",
            version=datasets.Version("1.0.0", ""),
            description="Tokenized FMA",
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
            citation=_CITATION
        )

    def _split_generators(self, dl_manager):
        print(self.config)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"data_file": f"{self.config.data_dir}/fma-train.txt"}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"data_file": f"{self.config.data_dir}/fma-small-valid.txt"}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"data_file": f"{self.config.data_dir}/fma-small-test.txt"}),
        ]

    def _generate_examples(self, data_file):
        """This function returns the FMA text in the discretized, tokenized form."""
        logger.info("generating examples from = %s", data_file)
        with open(data_file, encoding="utf-8") as f:
            for idx, row in enumerate(f):
                yield idx, {"text": row}

