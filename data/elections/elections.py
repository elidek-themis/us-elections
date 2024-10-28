import datasets

_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", 
    "Connecticut", "Delaware", "District of Columbia", "Florida", "Georgia", "Hawaii", 
    "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", 
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", 
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", 
    "New Hampshire", "New Jersey", "New Mexico", "New York", 
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", 
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", 
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", 
    "West Virginia", "Wisconsin", "Wyoming"
]

class ElectionPrompts(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="test",
            version=VERSION,
            description="US Elections prompts"
        )
    ]

    DEFAULT_CONFIG_NAME = "test"

    def _info(self):
        features = datasets.Features(
            {
                "state": datasets.Value("string"),
                "prompt": datasets.Value("string")
            }
        )

        return datasets.DatasetInfo(
            description="US Elections prompts",
            features=features
            )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"states": _STATES}
            )
        ]

    def _generate_examples(self, states):
        for i, state in enumerate(states):
            prompt = f"As a resident of {state}, in the 2024 US presidential election I will vote for the candidate of the"
            yield i, {"state":state, "prompt":prompt}