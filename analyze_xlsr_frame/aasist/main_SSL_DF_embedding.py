import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import yaml
from data_utils_SSL import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval
from model_embedding import Model
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from tqdm import tqdm
import pandas as pd

__author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"



def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    with torch.no_grad():
        for batch_x, batch_y in tqdm.tqdm(dev_loader, desc="Validation", leave=False):
            
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out = model(batch_x)
            
            batch_loss = criterion(batch_out, batch_y)
            val_loss += (batch_loss.item() * batch_size)
            
        val_loss /= num_total
   
    return val_loss


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)
    model.eval()
    
    fname_list = []
    score_list = []
    
    with torch.no_grad():
        for batch_x,utt_id in tqdm.tqdm(data_loader, desc="Evaluation", leave=False):
            batch_size = batch_x.size(0)
            batch_x = batch_x.to(device)
            batch_out = model(batch_x)
            fname_list = list(utt_id)
            score_list = batch_out.data.cpu().numpy().tolist()
            with open(save_path, 'a+') as fh:
                for f, cm in zip(fname_list, score_list):
                    fh.write('{} {} {}\n'.format(f, cm[0], cm[1]))
                    
    print('Scores saved to {}'.format(save_path))
    

def produce_embedding_file(dataset, model, device, embedding_save_path,
                           batch_size=64, utt_id_list=None):
    """
    Dev set 임베딩(x_ssl_feat) 추출 및 저장.
    - 모델 forward가 (logits, x_ssl_feat)를 반환한다고 가정
    - dataset이 (x, y)만 줄 경우 utt_id_list(=file_list)를 이용해 매핑
    """
    os.makedirs(embedding_save_path, exist_ok=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()

    # dataset에 list_IDs가 있으면 우선 사용, 없으면 인자로 받은 utt_id_list 사용
    list_ids = getattr(dataset, "list_IDs", None)
    if list_ids is None:
        list_ids = utt_id_list  # None일 수도 있음
    running_idx = 0

    records, n_saved = [], 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting Embeddings", ncols=100):
            # 배치 언패킹: (x, y) 또는 (x, y, utt_ids) 등
            if isinstance(batch, (list, tuple)):
                batch_x = batch[0]
                # utt_ids가 배치 안에 없으면 list_ids에서 순서대로 뽑기
                if len(batch) >= 3:
                    utt_ids = [str(u) for u in batch[2]]
                else:
                    bsz = batch_x.size(0)
                    if list_ids is None:
                        utt_ids = [f"utt_{running_idx+i:07d}" for i in range(bsz)]
                    else:
                        utt_ids = [str(list_ids[running_idx+i]) for i in range(bsz)]
                    running_idx += bsz
            else:
                raise ValueError("Unexpected batch structure from DataLoader.")

            batch_x = batch_x.to(device)
            logits, x_ssl_feat = model(batch_x)   # (logits, x_ssl_feat)
            embs = x_ssl_feat.detach().cpu().numpy()

            for uid, emb in zip(utt_ids, embs):
                base = os.path.splitext(os.path.basename(uid))[0]
                save_path = os.path.join(embedding_save_path, base + ".npy")
                i = 1
                while os.path.exists(save_path):
                    save_path = os.path.join(embedding_save_path, f"{base}_{i}.npy"); i += 1
                np.save(save_path, emb.astype(np.float32))
                records.append({"utt_id": uid, "embedding_path": save_path})
                n_saved += 1

    pd.DataFrame(records).to_csv(os.path.join(embedding_save_path, "embedding_index.csv"), index=False)
    print(f"[OK] Embeddings saved: {n_saved} → {embedding_save_path}")

def train_epoch(train_loader, model, lr,optim, device):
    running_loss = 0
    
    num_total = 0.0
    
    model.train()

    #set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    for batch_x, batch_y in tqdm.tqdm(train_loader, desc="Training"):
       
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        
        batch_loss = criterion(batch_out, batch_y)
        
        running_loss += (batch_loss.item() * batch_size)
       
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
       
    running_loss /= num_total
    
    return running_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    # Dataset
    parser.add_argument('--database_path', type=str, default='/your/path/to/data/ASVspoof_database/DF/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 DF for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 DF eval data folders are in the same database_path directory.') 
    '''
    % database_path/
    %   |- DF
    %      |- ASVspoof2021_DF_eval/flac
    %      |- ASVspoof2019_LA_train/flac
    %      |- ASVspoof2019_LA_dev/flac
    '''

    parser.add_argument('--protocols_path', type=str, default='database/', help='Change with path to user\'s DF database protocols directory address')
    '''
    % protocols_path/
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt
 
    %   |- ASVspoof_DF_cm_protocols
    %      |- ASVspoof2021.DF.cm.eval.trl.txt
  
    '''

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    # model
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    parser.add_argument('--early_stop_patience', type=int, default=5,
                        help='Early stopping patience (default: 5)')
    # Auxiliary arguments
    parser.add_argument('--track', type=str, default='DF',choices=['LA', 'PA','DF'], help='LA/PA/DF')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--is_eval', action='store_true', default=False,help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)')
    parser.add_argument('--dump_dev_embeddings', action='store_true',
                    help='dev set 임베딩 추출만 수행')
    parser.add_argument('--embedding_dir', type=str, default='dev_embeddings',
                    help='임베딩 저장 경로')


    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=3, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
    
    ##===================================================Rawboost data augmentation ======================================================================#
    

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
 
    #make experiment reproducible
    set_random_seed(args.seed, args)
    
    track = args.track

    assert track in ['LA', 'PA','DF'], 'Invalid track given'

    #database
    prefix_2021 = 'ASVspoof2021.{}'.format(track)
    
    #define model saving path
    model_tag = 'model_{}_{}_{}_{}_{}'.format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    model = Model(args,device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model =nn.DataParallel(model).to(device)
    print('nb_params:',nb_params)

    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path,map_location=device))
        print('Model loaded : {}'.format(args.model_path))


    #evaluation 
    if args.eval:
        file_eval = genSpoof_list( dir_meta = os.path.join(args.protocols_path),is_train=False,is_eval=True)
        print('no. of eval trials',len(file_eval))
        eval_set=Dataset_ASVspoof2021_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path))
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        sys.exit(0)
   
         
    # define train dataloader
    d_label_trn,file_train = genSpoof_list( dir_meta =  os.path.join(args.protocols_path),is_train=True,is_eval=False)
  
    print('no. of training trials',len(file_train))
    
    train_set=Dataset_ASVspoof2019_train(args, list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path),algo=args.algo)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=8, shuffle=True,drop_last = True)
    
    del train_set,d_label_trn
    

    # define dev (validation) dataloader

    d_label_dev,file_dev = genSpoof_list( dir_meta =  os.path.join(args.protocols_path),is_train=False,is_eval=False)
    
    print('no. of validation trials',len(file_dev))
    
    dev_set = Dataset_ASVspoof2019_train(args,list_IDs = file_dev,labels = d_label_dev,base_dir = os.path.join(args.database_path),algo=args.algo)

    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=8, shuffle=False)

    del dev_set,d_label_dev

    # >>> 여기에 추가 <<<
    if args.dump_dev_embeddings:
        produce_embedding_file(
            dataset=dev_loader.dataset,
            model=model,
            device=device,
            embedding_save_path=args.embedding_dir,
            batch_size=args.batch_size
        )
        sys.exit(0)
    

    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    patience = args.early_stop_patience
    
    for epoch in tqdm.tqdm(range(num_epochs), desc="Epochs"):
        
        running_loss = train_epoch(train_loader,model, args.lr,optimizer, device)
        val_loss = evaluate_accuracy(dev_loader, model, device)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print('\n{} - {} - {} '.format(epoch,
                                                   running_loss,val_loss))
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(
                model_save_path, 'best_model.pth'))
            print('New best validation loss: {:.6f} - Model saved!'.format(best_val_loss))
        else:
            patience_counter += 1
            print('No improvement. Patience: {}/{}'.format(patience_counter, patience))
        
        # Save epoch model (optional)
        torch.save(model.state_dict(), os.path.join(
            model_save_path, 'epoch_{}.pth'.format(epoch)))
        
        # Early stopping
        if patience_counter >= patience:
            print('Early stopping triggered after {} epochs without improvement'.format(patience))
            print('Best validation loss: {:.6f}'.format(best_val_loss))
            break
