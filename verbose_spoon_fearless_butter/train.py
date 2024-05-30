import os
import pickle
from typing import Callable, Iterable

import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import Progress, TaskID, TextColumn, BarColumn, TimeRemainingColumn, Column
from rich.prompt import Confirm
from rich.traceback import install as install_rich_traceback
from torch.optim import SGD, Adam, RMSprop
from torch.utils.data import DataLoader

from verbose_spoon_fearless_butter.model import Net, CSIDataset
from verbose_spoon_fearless_butter.util.fixed_task_progress import FixedTaskProgressColumn

console = Console()
error_console = Console(stderr=True, style="bold red")
# install_rich_traceback(console=error_console, show_locals=True, width=None, suppress=[torch, pickle])

TRAIN_FILE_NAME = 'dataset_0_unwrap.pkl'
EVAL_FILE_NAME = 'dataset_0_unwrap.pkl'
EPOCH_COUNT = 50
BATCH_SIZE = 30
DEVICE = torch.device('mps' if BATCH_SIZE >= 35 else 'cpu')
# DEVICE = torch.device('cpu')

model = Net()
prev_min_loss = float("inf")

if os.path.isfile('model.pth'):
    console.log('Found model.pth file')
    if Confirm.ask('Would you like to load the saved parameters?', console=console, default=True):
        with console.status('Loading saved parameters...', spinner='bouncingBall'):
            state = torch.load('model.pth')
            if 'previous_loss' in state:
                prev_min_loss = state.pop('previous_loss')
            model.load_state_dict(state)
        console.log('Loaded saved parameters')
    elif Confirm.ask('Would you like to delete the saved parameters?', console=console, default=True):
        os.remove('model.pth')
        console.log('Deleted saved parameters')

with console.status('Loading datasets...', spinner='bouncingBall') as status:
    training_dataset: CSIDataset = pickle.load(open(TRAIN_FILE_NAME, 'rb'))
    testing_dataset: CSIDataset = pickle.load(open(EVAL_FILE_NAME, 'rb'))  # ToDo: use the correct datatset (random subsrt of train: https://discuss.pytorch.org/t/how-to-test-model-with-unseen-data/121218)
    console.log('Loaded datasets')

    status.update('Setting up...')
    criterion = nn.PairwiseDistance(p=2)
    # optimizer = SGD(model.parameters(), lr=1e-10, momentum=0.9)
    # optimizer = SGD(model.parameters(), lr=1e-3)
    optimizer = Adam(model.parameters(), lr=1e-3)
    # optimizer = RMSprop(model.parameters(), lr=1e-2, momentum=0.9)

    training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=False)
    evaluation_dataloader = DataLoader(testing_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if DEVICE.type != 'cpu':
        status.update(f'Moving tensors to {DEVICE.type} device...')
        model = model.to(DEVICE)
        training_dataset.to(torch.float32)
        testing_dataset.to(torch.float32)
        training_dataset.to(DEVICE)
        testing_dataset.to(DEVICE)
        criterion = criterion.to(DEVICE)
        console.log(f'Moved tensors to {DEVICE.type} device')
console.log('Setup done')

def train(_model: nn.Module, dataloader: DataLoader, track: Callable[[Iterable], Iterable]) -> float:
    _model.train()
    _model.last_hidden = None
    _model.lstm_last_hidden = None

    counter = 0
    train_loss = torch.tensor(0.0, device=DEVICE)
    for input_, truth in track(dataloader):
        optimizer.zero_grad()

        prediction = _model(input_)

        loss = torch.mean(criterion(prediction, truth))
        loss.backward()

        optimizer.step()

        train_loss += loss
    train_loss /= len(dataloader)

    return train_loss.item()


def evaluate(_model: nn.Module, dataloader: DataLoader, track: Callable[[Iterable], Iterable]) -> float:
    _model.eval()
    _model.last_hidden = None
    _model.lstm_last_hidden = None

    counter = 0
    eval_loss = torch.tensor(0.0, device=DEVICE)
    with torch.no_grad():
        for input_, truth in track(dataloader):
            prediction = _model(input_)

            loss = torch.mean(criterion(prediction, truth))

            eval_loss += loss.item()
        eval_loss /= len(dataloader)

    return eval_loss.item()


min_eval_loss: float = prev_min_loss
if min_eval_loss != float('inf'):
    console.log(f'Loaded minimum loss: {min_eval_loss:.4f}')
columns = (
    TextColumn('[progress.description]{task.description}'),
    BarColumn(bar_width=None),
    FixedTaskProgressColumn(show_speed=True, table_column=Column(no_wrap=True, width=21, justify='left')),
    TimeRemainingColumn(compact=True, elapsed_when_finished=True)
)
console.rule('Training')
with (Progress(*columns, console=console) as progress):
    for epoch in progress.track(range(EPOCH_COUNT), description='Training...'):
        if epoch == 0:
            progress.update(TaskID(0), unit='epoch')
            train_task = progress.add_task('Training...',
                                           total=len(training_dataloader),
                                           unit='batch',
                                           start=False)
            eval_task = progress.add_task('Evaluating...',
                                          total=len(testing_dataset),
                                          unit='batch',
                                          start=False)
        progress.update(TaskID(0), description=f'Training, epoch [repr.number]{epoch}[default]...')
        try:
            progress.reset(eval_task, description='Evaluating...', visible=False, start=False)

            progress.reset(train_task, description='Training...')
            training_loss = train(model, training_dataloader, lambda seq: progress.track(seq, task_id=train_task))
            progress.update(train_task, description='Training [bold green]done')
            progress.stop_task(train_task)

            progress.reset(eval_task, description='Evaluating...', visible=True)
            evaluation_loss = evaluate(model, evaluation_dataloader, lambda seq: progress.track(seq, task_id=eval_task))
            # evaluation_loss = training_loss
            progress.update(eval_task, description='Evaluation [bold green]done')
            progress.stop_task(eval_task)

        except KeyboardInterrupt:
            console.log('Shutdown requested')
            break

        console.log(
            f'Finished epoch {epoch}, Training loss {training_loss:.4f}, Evaluation loss {evaluation_loss:.4f}'
        )

        if evaluation_loss < min_eval_loss:
            if min_eval_loss != float('inf'):
                console.log(f'Loss dropped from {min_eval_loss:.4f} to {evaluation_loss:.4f}, Saving model...')
            else:
                console.log('Saving model...')
            save_task = progress.add_task('Saving model...', total=None)

            min_eval_loss = evaluation_loss

            state = model.state_dict()
            state['previous_loss'] = min_eval_loss
            torch.save(state, 'model.pth')

            progress.remove_task(save_task)

console.rule('Done')
console.log(f'Training done, Loss: {min_eval_loss}')
