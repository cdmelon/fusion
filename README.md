

# Code README

Code and datasets for the COLING 2025 paper "Fusion meets Function: The Adaptive Selection-Generation Approach in Event Argument Extraction".
This codebase is based on PAIE (Prompting Argument Interaction for Event Argument Extraction), with modifications to allow for the combination of selective and generative methods.

## Usage

By default, running the `engine.py` file will use the BART-base model with the WikiEvent dataset. However, you can easily switch between different configurations as follows:

### Switching Models

To switch between BART-base and BART-large models, open the `config_parser.py` file and locate the `model_name_or_path` parameter.

```python
parser.add_argument("--model_name_or_path", default="./bart", type=str,
                        help="pre-trained language model")
```

### Switching Datasets

To switch between the RAMS and WikiEvent datasets, open the `config_parser.py` file and locate the `dataset_type` parameter. Set it to `"rams"` or `"wikievent"` as desired.

```python
parser.add_argument("--dataset_type", default="wikievent", choices=["rams", "wikievent"], type=str,
                        help="dataset type document-level(rams/wikievent)")
parser.add_argument("--role_path", default='./data/dset_meta/description_wikievent.csv', type=str, 
                    help="a file containing all role names. Read it to access all argument roles of this dataset")
parser.add_argument("--prompt_path", default='./data/prompts/prompts_wikievent_full.csv', type=str, 
                    help="a file containing all prompts we use for this dataset")
```

### Adjusting Selection and Generation Ratio

The model's behavior can be adjusted by modifying the `self.loss_ratio` parameter in the `models/paie.py` file. This parameter controls the ratio between selective and generative losses. Adjust it as needed to achieve the desired balance between selection and generation.

```python
class PAIEModel(nn.Module):
    def __init__(self):
        # Other model initialization code

        # Adjust the loss ratio to control selection vs. generation balance
        self.loss_ratio = 0.5  # Modify this value as needed
```

## Running the Code

Once you have configured the desired model, dataset, and loss ratio, you can run the code by executing `engine.py`. The code will use the specified settings to perform event arguments extraction.


