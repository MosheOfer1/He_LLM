import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

from custom_datasets.create_datasets import read_file_lines
import torch
from llm.gpt2_llm import GPT2LLM

from custom_datasets.seq2seq_dataset import Seq2SeqDataset
from translation.helsinki_translator import HelsinkiTranslator

from custom_transformers.transformer_1 import Transformer1


def main():
    import torch.distributed as dist

    # Check GPU availability
    gpu_count = torch.cuda.device_count()
    print(f"Process started - GPU count: {gpu_count}")
    if gpu_count == 0:
        print("No GPUs available. Exiting.")
        exit(1)

    # Check if we are in a distributed environment
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        # We are running under MPI via mpirun
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

        # Set environment variables
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

        # Set the master address and port
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '12355')

        # Initialize the process group
        print(f"Process {rank} - Initializing distributed process group...")
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://{master_addr}:{master_port}',
            world_size=world_size,
            rank=rank
        )
    else:
        # Single process (not distributed)
        world_size = 1
        rank = 0
        local_rank = 0
        os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
        print("Single process training - not using distributed training.")
        dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)

    # Set the device
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
    print(f"Process {rank} - local_rank: {local_rank}, CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"Process {rank} - Using device: {device}, Available GPUs: {torch.cuda.device_count()}")

    # Define local paths for the HPC
    translator1_local_path = "/home/management/scratch/talias/opus-mt-tc-big-he-en/"
    translator2_local_path = "/home/management/scratch/talias/opus-mt-en-he/"
    llm_local_path = "/home/management/scratch/dianab/polylm-1.7b/"
    text_file_path = "my_datasets/ynet_256k.txt"

    # Define default Hugging Face model names
    default_translator1_model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
    default_translator2_model_name = "Helsinki-NLP/opus-mt-en-he"
    default_llm_model_name = "facebook/opt-350m"

    # Check if the local paths exist, otherwise assign the Hugging Face model name
    translator1_model_name = translator1_local_path if os.path.exists(
        translator1_local_path) else default_translator1_model_name
    translator2_model_name = translator2_local_path if os.path.exists(
        translator2_local_path) else default_translator2_model_name
    llm_model_name = llm_local_path if os.path.exists(llm_local_path) else default_llm_model_name

    # Initialize models after setting the device
    print(f"Process {rank} - Initializing LLM on device {device}")
    llm = GPT2LLM(llm_model_name, device=device)

    print(f"Process {rank} - Initializing Translator on device {device}")
    translator = HelsinkiTranslator(
        translator1_model_name,
        translator2_model_name,
        device=device
    )

    print(f"Process {rank} - Initializing Transformer1 on device {device}")
    trans1 = Transformer1(
        translator,
        llm,
        device=device,
        nhead=2,
        num_layers=2
    )

    text = read_file_lines(text_file_path)

    print(f"Process {rank} - len(text) = {len(text)}")

    split_index = int(len(text) * 0.9)
    train_data, eval_data = text[:split_index], text[split_index:]
    print(f"Process {rank} - Train number of sentences = {len(train_data)}")
    print(f"Process {rank} - Eval number of sentences = {len(eval_data)}")

    # Create datasets
    print(f"Process {rank} - Creating datasets")
    train_dataset = Seq2SeqDataset(
        sentences=train_data,
        translator=translator,
        llm=llm,
    )

    eval_dataset = Seq2SeqDataset(
        sentences=eval_data,
        translator=translator,
        llm=llm,
    )

    print(f"Process {rank} - Starting training")
    trans1.train_model(train_dataset, eval_dataset)

    # Clean up
    print(f"Process {rank} - Destroying process group")
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
