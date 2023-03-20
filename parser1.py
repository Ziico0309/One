import argparse

parser1= argparse.ArgumentParser(description="Run selected model")
parser1.add_argument('-e','--epoch',type=int,default=1,help="Number of epochs")
parser1.add_argument('-b','--batch',type=int,default=256,help="Batch size")
parser1.add_argument('-lr', '--lr', default=1e-3, type=float,help='learning rate for optimizer')
parser1.add_argument('-dl','--download',type=str,default='True',help='Download or not')
parser1.add_argument('-k','--top_k',type=int,default=10,help='choose top@k for NDCG@k, HR@k')
parser1.add_argument('-fi','--file_size',type=str,default='100k',help='choose file size, [100k,1m,10m,20m]')
args = parser1.parse_args()