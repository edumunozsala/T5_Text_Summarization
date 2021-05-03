import argparse
import json
import sys
import sagemaker_containers

# Importing stock libraries
import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

# WandB – Import the wandb library
import wandb

from datasets import CustomDataset

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")
    pass


def _get_train_data_loader(tokenizer, training_dir, filename, batch_size, max_len, summ_len, nsamples, val_frac, seed=42):
    print("Get train and validation dataloaders.")
    # Importing and Pre-Processing the domain data
    # Selecting the needed columns only. 
    # Adding the summarzie text in front of the text. This is to format the dataset similar to how T5 model was trained for summarization task. 
    df = pd.read_csv(os.path.join(training_dir, filename), encoding='latin-1', nrows=nsamples)
    df = df[['text','ctext']]
    df.ctext = 'summarize: ' + df.ctext
    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest will be used for validation. 
    train_size = 1. - val_frac
    train_dataset=df.sample(frac=train_size,random_state = seed)
    val_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(val_dataset.shape))

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = CustomDataset(train_dataset, tokenizer, max_len, summ_len)
    val_set = CustomDataset(val_dataset, tokenizer, max_len, summ_len)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 0
        }

    val_params = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': 0
        }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    return training_loader, val_loader

# Creating the training function. This will be called in the main function. It is run depending on the epoch value.
# The model is put into train mode and then we wnumerate over the training loader and passed to the defined network 

def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for _,data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        #y_ids = y[:, :-1].contiguous() # Original
        # NO FUNCIONA y_ids = y[:, :].contiguous()
        #lm_labels = y[:, 1:].clone().detach() # Original
        lm_labels = y[:, :].clone().detach()
        # NO FUNCIONA lm_labels = y[:, :].clone().detach()
        #lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100 # Original
        lm_labels[y[:, :] == tokenizer.pad_token_id] = -100
        # NO FUNCIONA lm_labels[y[:, :] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        #outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
        #outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
        outputs = model(input_ids = ids, attention_mask = mask, labels=lm_labels)
        loss = outputs[0]
        
        if _%10 == 0:
            wandb.log({"Training Loss": loss.item()})

        if _%500==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # xm.optimizer_step(optimizer)
        # xm.mark_step()
        
# Mmodel validation on the validation daa¡taset 
def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=150, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            if _%100==0:
                print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals

if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--max-len', type=int, default=512, metavar='N',
                        help='input max sequence length for training (default: 512)')
    parser.add_argument('--summ-len', type=int, default=150, metavar='N',
                        help='summary max sequence length (default: 150)')
    parser.add_argument('--train_epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--val_epochs', type=int, default=1, metavar='N',
                        help='number of epochs to validate (default: 1)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                        help='learning rate to train (default: 0.01)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--val-frac', type=float, default=0.1, metavar='N',
                        help='Fraction og data for validation (default: 0.1)')
    parser.add_argument('--datafile', type=str, default='news_summary.csv', metavar='N',
                        help='Filename of the input data (default: news_summary.csv)')
    parser.add_argument('--nsamples', type=int, default=500, metavar='S',
                        help='Count of samples to train on (default: 500)')

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args = parser.parse_args()

    # WandB – Initialize a new run
    # Set the project name, the run name, the description
    wandb.init(project="pruebas_sagemaker", name="run-demo", 
               notes="Training demo of T5 transformer for the news summary dataset")
    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    # Defining some key variables that will be used later on in the training  
    config = wandb.config          # Initialize config
    config.TRAIN_BATCH_SIZE = args.batch_size    # input batch size for training (default: 64)
    config.VALID_BATCH_SIZE = args.batch_size    # input batch size for testing (default: 1000)
    config.TRAIN_EPOCHS = args.train_epochs        # number of epochs to train (default: 10)
    config.VAL_EPOCHS = args.val_epochs 
    config.LEARNING_RATE = args.lr    # learning rate (default: 0.01)
    config.SEED = args.seed               # random seed (default: 42)
    config.MAX_LEN = args.max_len
    config.SUMMARY_LEN = args.summ_len 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(args.seed) # pytorch random seed
    np.random.seed(args.seed) # numpy random seed
    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    training_loader,val_loader= _get_train_data_loader(tokenizer, args.data_dir, args.datafile, args.batch_size,
                                                       args.max_len, args.summ_len, 
                                                       args.nsamples, args.val_frac, args.seed)
    
    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary. 
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    print('Creating the pretrained model')
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model = model.to(device)

    # Defining the optimizer that will be used to tune the weights of the network in the training session. 
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=args.lr)
    print('Activating WandB tracking')
    # Log metrics with wandb
    wandb.watch(model, log="all")
    # Training loop
    print('Initiating Fine-Tuning for the model on our dataset')
    for epoch in range(args.train_epochs):
        train(epoch, tokenizer, model, device, training_loader, optimizer)
        
    # Validation loop and saving the resulting file with predictions and acutals in a dataframe.
    # Saving the dataframe as predictions.csv
    print('Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe')
    for epoch in range(args.val_epochs):
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        
    final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
    final_df.to_csv(os.path.abspath(os.path.join(args.output_data_dir,'predictions.csv')))
    print('Output Files generated for review')

    # Save the model
    model.save_pretrained(args.model_dir)
    
