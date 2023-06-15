import torch
from numpy import float32
from fast_pytorch_kmeans import KMeans
from tqdm import tqdm
from .utils import distance
from .network import StoSAutoencoder
from .cluster import CMeans


class DeepCMeams(torch.nn.Module):
    """
    Distinguish emotions in bio-signals

    Arguments:
        signal: numpy.array
            dimension nb observation by nb of signals
        
        original_signal: numpy.array
            Raw signal that serves as baseline
        
        epochs_pre: integer, optional
            Number of epochs for pre-training the auto-encoder (default is 30)
        
        epochs_train: integer, optional
            Number of epochs for training the deep-cluter algorithm (default is 30)
        
        actu_weight_each: integer, optional
            After how many training epochs to update the clustering algorithm (default is 1)
        
        batch_size: integer, optional
            Batch size of the gradient descent (default is 128)
        
        weight: float between 0 and 1, optional
            Weight to prioritise the clustering. If is 0, only the auto-encoder is trained. If is 1, only the clusturing is trained. (default is 0.8)
        
        nb_classes: integer, optional
            Number of classes of the kmeans algorithm (default is 2)
        
        seq_len: integer, optional
            The length of the each piece of signal whence to extract features (default is 20)
        
        embedding: integer, optional
            Number of features to extract (default is 128)
        
        decive: torch.device, optional
            The device where to run the script (default is None)

    Returns:
        labels_hat: numpy.array
            The label of each piece of signal. If nb_classes is 2, pseudo-labeling is automatic and 1 means relax, 2 means stress.
        
        stress_index: numpy.array
            Index indicating how much a piece of signal belongs to each cluster. 
            If *nb_classes* is 2, the index indicates how much a piece of signal belong to the stress cluster.
        
    """
    def __init__(self, signal, time=None, device=None):
        super(DeepCMeams, self).__init__()
        
        self.signal = signal
        self.time = time

        self.signal = self.signal.astype(float32)
        self.nb_features = self.signal.shape[1]

        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def run(self, nb_classes, seq_len, u_init=None, epochs_pre=1, epochs_train=1, batch_size=128, 
            embedding=128, lam=2, m=1.5, lr_pretrain=1e-3, lr_train=1e-3, sigma=None, gamma=1,  pre_train_cluster=False): 
        
        self.lam = lam
        self.m = m
        self.lr_pretrain = lr_pretrain
        self.lr_train = lr_train

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.nb_classes = nb_classes
        self.embedding = embedding

        self.sigma = sigma
        self.gamma = gamma

        self.epochs_pre = epochs_pre
        self.epochs_train = epochs_train

        self.pre_train_cluster = pre_train_cluster
        self.U = u_init

        if self.U is not None and self.pre_train_cluster:
            self.pre_train_cluster = True
        else:
            self.pre_train_cluster = False

        if self.U is None:
            _, u, _, _, _, _, _ = CMeans(data=self.signal.T, c=self.nb_classes, m=self.m, error=0.000, maxiter=1000)# to change
            self.U = torch.from_numpy(u.T).type(torch.FloatTensor)
            self.U = self.U[self.seq_len:].detach().to(self.device)
        else:
            self.U = torch.from_numpy(self.U).type(torch.FloatTensor)
            self.U = self.U[self.seq_len:].detach().to(self.device)          
        

        data = [(self.signal[i-self.seq_len:i+1,:], i-self.seq_len) for i in range(self.seq_len, self.signal.shape[0])]
        self.dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)

        ### The model
        self.AE = StoSAutoencoder(seq_len=self.seq_len, n_features=self.nb_features, embedding_dim=self.embedding, hidden_dim=128)
        self.AE = self.AE.to(self.device)
    
        self.memory_loss = {"fcm":[], "r":[]}

        ## pre-train auto-encoder
        self.init_centroids()

        with tqdm(range(self.epochs_pre), unit="epoch", desc="Pre-training") as tepoch:
            for _ in tepoch:
                if self.pre_train_cluster:
                    self._update_D()
                self.pre_train()
                if self.pre_train_cluster:
                    self._update_centroids()
                tepoch.set_postfix(loss=round(self.memory_loss["r"][-1], 2))

        print("init range", self.U.max() -self.U.min())
        
        ## Mix training
        self._update_centroids()
        with tqdm(range(self.epochs_train), unit="epoch", desc="Training") as tepoch:
            for _ in tepoch:
                self._update_D()
                self.train()
                self.clustering()
                L = (round(self.memory_loss["r"][-1], 2), round(self.memory_loss["fcm"][-1], 2))
                print("range", self.U.max() -self.U.min())
                tepoch.set_postfix(loss=L)


        ## results
        labels_hat = torch.argmax(self.U, dim=1) +1
        if self.time is None:
            return labels_hat.cpu().detach().numpy(), self.U.cpu().detach().numpy()
        else:
            return labels_hat.cpu().detach().numpy(), self.U.cpu().detach().numpy(), self.time[self.seq_len:]

    def pre_train(self):
        optimizer_AE = torch.optim.Adam(self.AE.parameters(), lr=self.lr_pretrain, weight_decay=self.lr_pretrain*1e-1)

        self.Z = []
        L = 0
        for batch in self.dataloader:
            x, indices = batch
            x = x.to(self.device)

            x_train = x[:,0:-1,:]
            x_obj = x[:,1:,:]

            x_hat, z = self.AE(x_train)
            
            z = torch.flatten(z, 1, 2)
            self.Z.append(z)

            if self.pre_train_cluster:
                u = self.U[indices]
                d = self.D[indices]
            
            optimizer_AE.zero_grad()
            loss = self.loss_ae(x_obj, x_hat)
            L += loss.item()
            if self.pre_train_cluster:
                loss += (self.lam/2) * self.loss_fcm(z, d, u)
            loss.backward()
            optimizer_AE.step()
            

        self.memory_loss["r"].append(L)
            
        self.Z = torch.cat(self.Z, 0)


    def train(self):
        optimizer_AE = torch.optim.Adam(self.AE.parameters(), lr=self.lr_train, weight_decay=self.lr_train*1e-1)
        
        self.Z = []
        L_ae = 0
        L_fcm = 0
        for batch in self.dataloader:
            x, indices = batch
            x = x.to(self.device)
            x_train = x[:,0:-1,:]
            x_obj = x[:,1:,:]

            x_hat, z = self.AE(x_train)
            
            z = torch.flatten(z, 1, 2)
            self.Z.append(z)
            
            u = self.U[indices]
            d = self.D[indices]

            optimizer_AE.zero_grad()
            loss_ae = self.loss_ae(x_obj, x_hat)
            loss_fcm = self.loss_fcm(z, d, u)
            loss = loss_ae + (self.lam / 2) * loss_fcm
            loss.backward()
            optimizer_AE.step()
            L_ae += loss_ae.item()
            L_fcm += loss_fcm.item()

        self.memory_loss["r"].append(L_ae)
        self.memory_loss["fcm"].append(L_fcm)
            
        self.Z = torch.cat(self.Z, 0)


    def init_centroids(self):
        self.centroids = torch.rand(self.embedding, self.nb_classes, device=self.device).detach()

    def _update_U(self):
        if self.sigma is None:
            distances = distance(self.Z, self.centroids, False)
        else:
            distances = adaptive_loss(distance(self.Z, self.centroids, False), self.sigma)
        
        U = torch.exp(-distances / self.gamma)
        
        U = U / U.sum(dim=1).reshape([-1, 1])
        self.U = U.detach()


    def _update_D(self):
        if self.sigma is None:
            D = torch.ones([self.signal.shape[0] -self.seq_len, self.centroids.size(1)]).to(self.device)
        else:
            try:
                distances = distance(self.Z, self.centroids, False)
                D = (1 + self.sigma) * (distances + 2 * self.sigma) / (2 * (distances + self.sigma))
            except:
                D = torch.ones([self.signal.shape[0] -self.seq_len, self.centroids.size(1)]).to(self.device)
        self.D = D.detach()

    def _update_centroids(self):
        self._update_D()
        T = self.D * self.U

        centroids = self.Z.t().matmul(T) / T.sum(dim=0).reshape([1, -1])
        self.centroids = centroids.detach()

    def clustering(self, max_iter=1):
        for i in range(max_iter):
            self._update_centroids()    
            self._update_U()


    def loss_ae(self, x, recons_x):
        size = x.size(0)
        loss_ae = 1/2 * torch.norm(x - recons_x, p='fro') ** 2 / size
        for _, layer in self.AE.named_modules():
            if isinstance(layer, torch.nn.GRU):
                for _, weight in layer.named_parameters():
                    loss_ae += 10**-5 * (weight.norm()**2) / size
            elif isinstance(layer, torch.nn.Linear):
                loss_ae += 10**-5 * (layer.weight.norm()**2 + layer.bias.norm()**2) / size
            
        return loss_ae

    def loss_fcm(self, z, d, u):
        size = z.size(0)
        t = d*u

        distances = distance(z, self.centroids)
        distances = distances.type(torch.FloatTensor)
        t = t.type(torch.FloatTensor)
        loss_fcm = torch.trace(distances.t().matmul(t)) / size
        return loss_fcm




def adaptive_loss(D, sigma):
    return (1 + sigma) * D * D / (D + sigma)



class DeepKMeans(torch.nn.Module):
    """
    Distinguish emotions in bio-signals

    Arguments:
        signal: numpy.array
            dimension nb observation by nb of signals
        
        original_signal: numpy.array
            Raw signal that serves as baseline
        
        epochs_pre: integer, optional
            Number of epochs for pre-training the auto-encoder (default is 30)
        
        epochs_train: integer, optional
            Number of epochs for training the deep-cluter algorithm (default is 30)
        
        actu_weight_each: integer, optional
            After how many training epochs to update the clustering algorithm (default is 1)
        
        batch_size: integer, optional
            Batch size of the gradient descent (default is 128)
        
        weight: float between 0 and 1, optional
            Weight to prioritise the clustering. If is 0, only the auto-encoder is trained. If is 1, only the clusturing is trained. (default is 0.8)
        
        nb_classes: integer, optional
            Number of classes of the kmeans algorithm (default is 2)
        
        seq_len: integer, optional
            The length of the each piece of signal whence to extract features (default is 20)
        
        embedding: integer, optional
            Number of features to extract (default is 128)
        
        decive: torch.device, optional
            The device where to run the script (default is None)

    Returns:
        labels_hat: numpy.array
            The label of each piece of signal. If nb_classes is 2, pseudo-labeling is automatic and 1 means relax, 2 means stress.
        
        stress_index: numpy.array
            Index indicating how much a piece of signal belongs to each cluster. 
            If *nb_classes* is 2, the index indicates how much a piece of signal belong to the stress cluster.
        
    """
    def __init__(self, signal, time=None, device=None):
        super(DeepKMeans, self).__init__()

        self.signal = signal
        self.time = time

        self.signal = self.signal.astype(float32)
        self.nb_features = self.signal.shape[1]

        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
	
    def run(self, nb_classes, seq_len, s_init=None, epochs_pre=1, epochs_train=1, batch_size=128, 
            embedding=30, lam=2, lr_pretrain=1e-3, lr_train=1e-3, pre_train_cluster=False): 
        
        self.lam = lam
        self.lr_pretrain = lr_pretrain
        self.lr_train = lr_train

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.nb_classes = nb_classes
        self.embedding = embedding

        self.epochs_pre = epochs_pre
        self.epochs_train = epochs_train

        self.pre_train_cluster = pre_train_cluster

        self.S = s_init

        if self.S is not None and self.pre_train_cluster:
            self.pre_train_cluster = True
        else:
            self.pre_train_cluster = False

        if self.S is None:
            kmeans = KMeans(n_clusters=nb_classes, mode='euclidean', verbose=0, max_iter=200)
            labels_pred = kmeans.fit_predict(torch.from_numpy(self.signal))
            cluster_centers = kmeans.centroids
            self.centroids = cluster_centers.t()
            self.centroids = self.centroids.detach()
            self.centroids = self.centroids.type(torch.FloatTensor)
            self.centroids = self.centroids.to(self.device)

            self.S = torch.nn.functional.one_hot(labels_pred, num_classes= nb_classes)
            self.S = self.S.type(torch.FloatTensor)
            self.S = self.S[self.seq_len:].detach().to(self.device)
        else:
            self.S = torch.from_numpy(self.S)
            self.S = torch.nn.functional.one_hot(self.S -1, num_classes= nb_classes)
            self.S = self.S.type(torch.FloatTensor)
            self.S = self.S[self.seq_len:].detach().to(self.device)          

        data = [(self.signal[i-self.seq_len:i+1,:], i-self.seq_len) for i in range(self.seq_len, self.signal.shape[0])]
        self.dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

        ### The model
        self.AE = StoSAutoencoder(seq_len=self.seq_len, n_features=self.nb_features, embedding_dim=self.embedding, hidden_dim=128)
        self.AE = self.AE.to(self.device)

        self.memory_loss = {"cluster":[], "r":[]}

        ## pre-train auto-encoder
        self.init_centroids()

        with tqdm(range(self.epochs_pre), unit="epoch", desc="Pre-training") as tepoch:
            for _ in tepoch:
                self.pre_train()
                if self.pre_train_cluster:
                    self._update_centroids()
                tepoch.set_postfix(loss=round(self.memory_loss["r"][-1], 2))


        ## Mix training
        self._update_centroids()
        with tqdm(range(self.epochs_train), unit="epoch", desc="Training") as tepoch:
            for _ in tepoch:
                self.train()
                self.clustering()
                L = (round(self.memory_loss["r"][-1], 2), round(self.memory_loss["cluster"][-1], 2))
                tepoch.set_postfix(loss=L)


        ## results
        labels_hat = torch.argmax(self.S, dim=1) +1
        if self.time is None:
            return labels_hat.cpu().detach().numpy()
        else:
            return labels_hat.cpu().detach().numpy(), self.time[self.seq_len:]
	

    def pre_train(self):
        optimizer_AE = torch.optim.Adam(self.AE.parameters(), lr=self.lr_pretrain, weight_decay=self.lr_pretrain*1e-1)

        self.Z = []
        L = 0
        for batch in self.dataloader:
            x, indices = batch
            x = x.to(self.device)
            x_train = x[:,0:-1,:]
            
            x_obj = x[:,1:,:]

            x_hat, z = self.AE(x_train)
            
            z = torch.flatten(z, 1, 2)
            self.Z.append(z)

            s = self.S[indices]

            optimizer_AE.zero_grad()
            loss = self.loss_ae(x_obj, x_hat)
            L += loss.item()
            if self.pre_train_cluster:
                loss += (self.lam / 2) * self.loss_kmeans(z, s)
            loss.backward()
            optimizer_AE.step()
            

        self.memory_loss["r"].append(L)

        self.Z = torch.cat(self.Z, 0)


    def train(self):
        optimizer_AE = torch.optim.Adam(self.AE.parameters(), lr=self.lr_train, weight_decay=self.lr_train*1e-1)

        self.Z = []
        L_ae = 0
        L_cluster = 0
        for batch in self.dataloader:
            x, indices = batch
            x = x.to(self.device)
            x_train = x[:,0:-1,:]
            x_obj = x[:,1:,:]

            x_hat, z = self.AE(x_train)

            z = torch.flatten(z, 1, 2)
            self.Z.append(z)

            s = self.S[indices]

            optimizer_AE.zero_grad()
            loss_ae = self.loss_ae(x_obj, x_hat)
            loss_cluster = self.loss_kmeans(z, s)
            loss = loss_ae + (self.lam / 2) * loss_cluster
            loss.backward()
            optimizer_AE.step()
            L_ae += loss_ae.item()
            L_cluster += loss_cluster.item()

        self.memory_loss["r"].append(L_ae)
        self.memory_loss["cluster"].append(L_cluster)

        self.Z = torch.cat(self.Z, 0)
        print("there is nan", torch.isnan(self.Z).any())


    def init_centroids(self):
        self.centroids = torch.rand(self.embedding, self.nb_classes, device=self.device).detach()


    def _update_centroids(self):
        for k in range(self.nb_classes):
            self.centroids[:,k] = torch.mean(self.Z[self.S.type(torch.BoolTensor)[:,k]],0)

    def _update_S(self):
        self.S = torch.zeros(self.Z.size(0), self.nb_classes)
        for k in range(self.nb_classes):
            self.S[:,k] = torch.linalg.norm(self.Z -self.centroids[:,k], dim=1)

        self.S = torch.argmin(self.S, dim=1)
        self.S = torch.nn.functional.one_hot(self.S, num_classes= self.nb_classes)
        self.S = self.S.type(torch.float)
        self.S = self.S.to(self.device)

    def clustering(self, max_iter=1):
        for i in range(max_iter):
            self._update_S()
            self._update_centroids()    


    def loss_ae(self, x, recons_x):
        size = x.size(0)
        loss_ae = 1/2 * torch.norm(x - recons_x, p='fro') ** 2 / size # may try something else, like MSE
        for _, layer in self.AE.named_modules():
            if isinstance(layer, torch.nn.GRU):
                for _, weight in layer.named_parameters():
                    loss_ae += 10**-5 * (weight.norm()**2) / size
            elif isinstance(layer, torch.nn.Linear):
                loss_ae += 10**-5 * (layer.weight.norm()**2 + layer.bias.norm()**2) / size
            
        return loss_ae

    def loss_kmeans(self, z, s):
        prod = torch.matmul(self.centroids, s.t()).t()
        prod = prod.type(torch.FloatTensor)
        loss = (z -prod.detach())**2
        loss = loss.mean()
        return loss
    
    def membership(self, m=2):
        D = torch.zeros(self.S.size()).to(self.device)
        for k in range(self.nb_classes):
            D[:,k] = torch.pow(torch.linalg.norm(self.Z -self.centroids[:,k], dim=1), 2/(1 -m))
        
        U = torch.zeros(self.S.size()).to(self.device)
        for k in range(self.nb_classes):
            U[:,k] = D[:,k]/torch.sum(D, dim=1)
        U = U.cpu().detach().numpy()
        return U
