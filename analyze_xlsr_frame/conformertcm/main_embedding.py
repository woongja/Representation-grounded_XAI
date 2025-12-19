import argparse
import sys
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from data_utils import Dataset_train, Dataset_eval
from model_embedding import Model
from utils import reproducibility
from utils import read_metadata
import numpy as np
from tqdm import tqdm
import pandas as pd

def evaluate_accuracy(dev_loader, model, device, epoch=0):
    val_loss = 0.0
    num_total = 0.0
    correct = 0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    with torch.no_grad():
        for batch_x, batch_y in tqdm(dev_loader, desc=f"Eval Epoch {epoch}", leave=False):
            batch_size = batch_x.size(0)
            target = torch.LongTensor(batch_y).to(device)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            batch_out, _ = model(batch_x)
            pred = batch_out.max(1)[1]
            correct += pred.eq(target).sum().item()

            batch_loss = criterion(batch_out, batch_y)
            val_loss += (batch_loss.item() * batch_size)

    val_loss /= num_total
    test_accuracy = 100. * correct / len(dev_loader.dataset)
    print('\n{} - {} - {} '.format(epoch, str(test_accuracy)+'%', val_loss))
    return val_loss


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)
    model.eval()
    fname_list = []
    score_list = []
    text_list = []
    
    with torch.no_grad():
        for batch_x,utt_id in tqdm(data_loader, desc="Evaluating", unit="batch"):
            batch_x = batch_x.to(device)
            batch_out, _ = model(batch_x)
            batch_score = (batch_out[:, 1]
                        ).data.cpu().numpy().ravel()
            # add outputs
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())

        for f, cm in zip(fname_list, score_list):
            text_list.append('{} {}'.format(f, cm))
        del fname_list
        del score_list
        with open(save_path, 'a+') as fh:
            for i in range(0, len(text_list), 500):
                batch = text_list[i:i+500]
                fh.write('\n'.join(batch) + '\n')
        del text_list
        fh.close()
        print('Scores saved to {}'.format(save_path))

import os, numpy as np, torch, pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

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
            logits,attn_score,  x_ssl_feat = model(batch_x)   # (logits, x_ssl_feat)
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
    num_total = 0.0
    model.train()

    #set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_batch = len(train_loader)
    i=0
    pbar = tqdm(train_loader)
    for i, batch in enumerate(pbar):
        batch_x, batch_y = batch

        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out, _ = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)     
        pbar.set_description(f"Epoch {epoch}: cls_loss {batch_loss.item()}")
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        i=i+1
    sys.stdout.flush()
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conformer-W2V')
    # Dataset
    parser.add_argument('--database_path', type=str, default='ASVspoof_database/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.')
    '''
    % database_path/
    %      |- ASVspoof2021_LA_eval/wav
    %      |- ASVspoof2019_LA_train/wav
    %      |- ASVspoof2019_LA_dev/wav
    %      |- ASVspoof2021_DF_eval/wav
    '''

    parser.add_argument('--protocols_path', type=str, default='ASVspoof_database/', help='Change with path to user\'s LA database protocols directory address')
    '''
    % protocols_path/
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt
  
    '''

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=7)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='WCE')

    #model parameters
    parser.add_argument('--emb-size', type=int, default=144, metavar='N',
                    help='embedding size')
    parser.add_argument('--heads', type=int, default=4, metavar='N',
                    help='heads of the conformer encoder')
    parser.add_argument('--kernel_size', type=int, default=31, metavar='N',
                    help='kernel size conv module')
    parser.add_argument('--num_encoders', type=int, default=4, metavar='N',
                    help='number of encoders of the conformer')
    parser.add_argument('--FT_W2V', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to fine-tune the W2V or not')
    
    # model save path
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    parser.add_argument('--comment_eval', type=str, default=None,
                        help='Comment to describe the saved scores')
    parser.add_argument('--early_stop_patience', type=int, default=3,
                        help='Early stopping patience (default: 3)')
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    
    #Train
    parser.add_argument('--train', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to train the model')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to model checkpoint for resuming training or evaluation')
    #Eval
    parser.add_argument('--n_mejores_loss', type=int, default=5, help='save the n-best models')
    parser.add_argument('--average_model', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether average the weight of the n_best epochs')
    parser.add_argument('--n_average_model', default=5, type=int)

    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=5, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

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
    parser.add_argument('--eval_output', type=str, default=None, help='Path to save the evaluation result')

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
    parser.add_argument('--dump_dev_embeddings', action='store_true',
                    help='dev set 임베딩 추출만 수행')
    parser.add_argument('--embedding_dir', type=str, default='dev_embeddings',
                    help='임베딩 저장 경로')
    
    ##===================================================Rawboost data augmentation ======================================================================#
    

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
    print(args)
    args.track='LA'
 
    #make experiment reproducible
    reproducibility(args.seed, args)
    
    track = args.track
    n_mejores=args.n_mejores_loss

    assert track in ['LA','DF'], 'Invalid track given'
    assert args.n_average_model<args.n_mejores_loss+1, 'average models must be smaller or equal to number of saved epochs'

    #database
    prefix      = 'ASVspoof_{}'.format(track)
    prefix_2019 = 'ASVspoof2019.{}'.format(track)
    prefix_2021 = 'ASVspoof2021.{}'.format(track)
    
    #define model saving path
    model_tag = 'Conformer_w_TCM_{}_{}_{}_ES{}_H{}_NE{}_KS{}_AUG{}_w_sin_pos'.format(
        track, args.loss, args.lr,args.emb_size, args.heads, args.num_encoders, args.kernel_size, args.algo)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models/FT')
    
    print('Model tag: '+ model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    best_save_path = os.path.join(model_save_path, 'best')
    if not os.path.exists(best_save_path):
        os.mkdir(best_save_path)
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    model = Model(args,device)
    # model.load_state_dict(torch.load(args.ckpt_path))
    model.load_state_dict(torch.load(args.model_path))
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters() if param.requires_grad])
    model =model.to(device)
    print('nb_params:',nb_params)

    
    # print('######## Eval ########')
    # if args.average_model:
    #     sdl=[]
    #     model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
    #     print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
    #     sd = model.state_dict()
    #     for i in range(1,args.n_average_model):
    #         model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
    #         print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
    #         sd2 = model.state_dict()
    #         for key in sd:
    #             sd[key]=(sd[key]+sd2[key])
    #     for key in sd:
    #         sd[key]=(sd[key])/args.n_average_model
    #     model.load_state_dict(sd)
    #     torch.save(model.state_dict(), os.path.join(best_save_path, 'avg_5_best_{}.pth'.format(i)))
    #     print('Model loaded average of {} best models in {}'.format(args.n_average_model, best_save_path))
    # else:
    #     model.load_state_dict(torch.load(os.path.join(args.model_path)))
    #     print('Model loaded : {}'.format(os.path.join(args.model_path)))

    # file_eval = read_metadata( dir_meta =  os.path.join(args.protocols_path), is_eval=True, is_train=False)
    # print('no. of eval trials',len(file_eval))
    # eval_set=Dataset_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path),track='LA')
    # produce_evaluation_file(eval_set, model, device, args.eval_output)
    # sys.exit()
    
    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
     
    # define train dataloader
    label_trn, files_id_train = read_metadata( dir_meta =  os.path.join(args.protocols_path), is_eval=False, is_train=True)
    print('no. of training trials',len(files_id_train))
    
    train_set=Dataset_train(args,list_IDs = files_id_train,labels = label_trn,base_dir = os.path.join(args.database_path),algo=args.algo)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers = 10, shuffle=True,drop_last = True)
    
    del train_set, label_trn
    
    # define validation dataloader
    labels_dev, files_id_dev = read_metadata( dir_meta =  os.path.join(args.protocols_path), is_eval=False, is_train=False)
    print('no. of validation trials',len(files_id_dev))

    dev_set = Dataset_train(args,list_IDs = files_id_dev,
		    labels = labels_dev,
		    base_dir = os.path.join(args.database_path), algo=args.algo)

    dev_loader = DataLoader(dev_set, batch_size=8, num_workers=10, shuffle=False)
    del dev_set,labels_dev

    
    if args.dump_dev_embeddings:
        produce_embedding_file(
            dataset=dev_loader.dataset,
            model=model,
            device=device,
            embedding_save_path=args.embedding_dir,
            batch_size=args.batch_size,
        )
    sys.exit(0)
    
    ##################### Training and validation #####################
    num_epochs = args.num_epochs
    epoch=0
    bests=np.ones(n_mejores,dtype=float)*float('inf')
    best_loss=float('inf')
    
    # Early stopping variables
    patience_counter = 0
    patience = args.early_stop_patience
    
    if args.train:
        for i in range(n_mejores):
            np.savetxt( os.path.join(best_save_path, 'best_{}.pth'.format(i)), np.array((0,0)))
        while patience_counter < patience:
            print('######## Epoca {} ########'.format(epoch))
            train_epoch(train_loader, model, args.lr, optimizer, device)
            val_loss = evaluate_accuracy(dev_loader, model, device)
            
            # Early stopping check and save only improved models
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(model_save_path, 'best.pth'))
                print('New best validation loss: {:.6f} - Model saved!'.format(best_loss))
            else:
                patience_counter += 1
                print('No improvement. Patience: {}/{}'.format(patience_counter, patience))
            for i in range(n_mejores):
                if bests[i]>val_loss:
                    for t in range(n_mejores-1,i,-1):
                        bests[t]=bests[t-1]
                        os.system('mv {}/best_{}.pth {}/best_{}.pth'.format(best_save_path, t-1, best_save_path, t))
                    bests[i]=val_loss
                    torch.save(model.state_dict(), os.path.join(best_save_path, 'best_{}.pth'.format(i)))
                    break
            print('\n{} - {}'.format(epoch, val_loss))
            print('n-best loss:', bests)
            #torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
            epoch+=1
            if epoch>74:
                break
            
            # Early stopping
            if patience_counter >= patience:
                print('Early stopping triggered after {} epochs without improvement'.format(patience))
                print('Best validation loss: {:.6f}'.format(best_loss))
                break
        print('Total epochs: ' + str(epoch) +'\n')

