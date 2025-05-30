import numpy as cp
try:
    import cupy
    if cupy.cuda.is_available():
        cp = cupy
except:
    pass
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import time

class RBM:
    def __init__(self, model):
        self.N = model['N']                     # number of visible units
        self.M = model['M']                     # number of hidden units
        self.eta = model['eta']                 # learning rate of the RBM
        self.batch_size = model['batch_size']   # batch size
        self.W_init = model['W_init']           #{'lecun', 'std'} --weight initialization
        
        if self.W_init == 'std':
            self.W = cp.random.normal(0.0, 1.0, (self.N, self.M))       # std
        else:
            self.W = cp.random.randn(self.N, self.M) / cp.sqrt(self.N)  # lecun
        
        self.a = cp.zeros((self.N))
        self.b = cp.zeros((self.M))
        self.persistent_hidden_states = cp.random.randint(0, 2, size=(self.batch_size,self.M))

        self.t_mse = []
        self.t_pnl = []
        self.t_ce = []
        self.v_mse = []
        self.v_pnl = []
        self.v_ce = []
        self.v_ssim = []
        
    def ssim_m(self,x1,x2):
        '''
        input : (B x N)
        output : scaler
        '''
        try:
            if cp == cupy:
                x1 = cp.asnumpy(x1)
                x2 = cp.asnumpy(x2)
        except:
            pass

        m = []
        for i in range(x1.shape[0]):
            m.append(ssim(x1[i], x2[i], data_range=1))
        
        m = cp.array(m)
        return m.mean()
    
    def v_energy(self,v):
        '''
        input : (B x N)
        output : (B)
        '''
        v = cp.array(v)
        wv_b = v.dot(self.W) + self.b
        return - v.dot(self.a.T) - cp.sum(cp.log(1.+ cp.exp(wv_b)),axis=1)
    
    def reconstruct(self,v_input,chains=1):
        '''
        input : (B x N)
        output : (B x N)
        '''
        v = cp.array(v_input)
        for _ in range(chains):
            h = self.sample_h_given_v(v)
            v = self.sample_v_given_h(h)
        return v
    
    def reconst_error(self,v_input,chains=1):
        '''
        input : (B x N)
        output : scaler
        '''
        v = self.reconstruct(v_input)
        mse = cp.mean(cp.sum((v_input - v)**2, axis=1), axis=0) # mean squared error
        return mse 
    
    def pseudo_neg_log_likelihood(self,batch_data):
        """return mean of pseudo-likelihood
        input : (B x N)
        output : scaler
        """
        idx_raw = cp.arange(batch_data.shape[0])
        idx_col = cp.random.randint(0,batch_data.shape[1],batch_data.shape[0])
        v_ = batch_data.copy()
        v_[idx_raw,idx_col] = 1 - v_[idx_raw,idx_col]
        e = self.v_energy(batch_data)
        e_ = self.v_energy(v_)
        L_ = -self.N * cp.log(self.activation((e_-e)))
        return cp.mean(L_)

    def cross_entropy(self,batch_data):
        """return mean of cross entropy
        input : (B x N)
        output : scaler
        """
        epss = 1e-6
        batch_data = cp.array(batch_data)
        h = self.sample_h_given_v(batch_data)
        p = self.activation(h.dot(self.W.T) + self.a)
        p = cp.clip(p, epss, 1. - epss)
        batch_data = batch_data
        m_H = batch_data*cp.log(p) + (1-batch_data)*cp.log(1.- p)
        return -cp.mean(m_H)

    def plot_samples(self, batch_data):
        n_cl = 10
        size = batch_data.shape[0] if batch_data.shape[0] < n_cl else n_cl
        idx = cp.random.choice(cp.arange(batch_data.shape[0]), size=size,replace=False)
        X_batch = batch_data[idx, :]
        v_eq = self.reconstruct(X_batch)
        X_batch = X_batch.reshape(size, int(self.N**0.5), int(self.N**0.5))
        try:
            if cp == cupy:
                X_batch = cp.asnumpy(X_batch)
                v_eq = cp.asnumpy(v_eq)
        except:
            pass
        plt.figure(figsize=(size,1))
        for i,im in enumerate(X_batch):
            ax=plt.subplot(1,size,i+1)
            ax.imshow(im)
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        v_eq = v_eq.reshape(size, int(self.N**0.5), int(self.N**0.5))
        plt.figure(figsize=(size,1))
        for i,im in enumerate(v_eq):
            ax=plt.subplot(1,size,i+1)
            ax.imshow(im)
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def plot_weights_hid(self,title='Receptive Fields'):
        '''
        Receptive fields.
        '''
        L_h = int((self.M)**(0.5))
        L_v = int((self.N)**(0.5))
        plt.clf()
        fig, axes = plt.subplots(L_h, L_h, gridspec_kw = {'wspace':0.1, 'hspace':0.1}, figsize=(8, 8))
        #fig.suptitle(title)
        try:
            if cp==cupy:
                W = cp.asnumpy(self.W)
        except:
            W = self.W
        for i in range(L_h):
            for j in range(L_h):
                axes[i, j].imshow(W[:,i*L_h+j].reshape(L_v, L_v), cmap='jet')
                axes[i, j].axis('off')
        plt.show()
        
    def activation(self, z):
        return 1 / (1 + cp.exp(-z))        

    def sample_h_given_v(self, v):
        '''
        input : (B x N)
        output : (B x M)
        '''
        prob = self.activation(v.dot(self.W) + self.b)
        h = cp.random.rand(v.shape[0], self.M) < prob
        return h.astype(float)

    def sample_v_given_h(self, h):
        '''
        input : (B x M)
        output : (B x N)
        '''
        prob = self.activation(h.dot(self.W.T) + self.a)
        v = cp.random.rand(h.shape[0], self.N) < prob
        return v.astype(float)

    def Gibbs_sampling(self, v_init, k, training):
        '''
        input : (B x N)
        output : ((B x N), (B x M))
        '''
        if training == 'CD':
            v = v_init
            for step in range(k):
                h = self.sample_h_given_v(v)
                v = self.sample_v_given_h(h)
        if training == 'PCD': 
            for step in range(k):
                v = self.sample_v_given_h(self.persistent_hidden_states)
                self.persistent_hidden_states = self.sample_h_given_v(v)
            h = self.sample_h_given_v(v)
        
        return v, h
            
    def minibatching_gradient_descent(self, batch_data, k, training, epoch, epochs):
        num_samples = batch_data.shape[0]
        
        # Calculate positive phase
        h_data = self.activation(batch_data.dot(self.W) + self.b)  # mean value of the hidden activations of the given batch data

        # Calculate negative phase
        v, h = self.Gibbs_sampling(batch_data, k, training)
        
        # Calculate mean hidden activations
        v_mean = v 
        h_mean = self.activation(v.dot(self.W) + self.b)

        # Update weights, biases, and hidden biases
        DW_p = cp.dot(batch_data.T, h_data) / num_samples
        DW_n = cp.dot(v_mean.T, h_mean) / num_samples
        DW   = DW_p - DW_n
        self.W = self.W + self.eta * DW

        Da = cp.mean(batch_data - v_mean, axis=0)
        self.a = self.a + self.eta * Da

        Db = cp.mean(h_data - h_mean, axis=0)
        self.b = self.b + self.eta * Db
            
        return DW, DW_n, DW_p, Da, Db

    def KH_update(self,batch_data,epoch,epochs,R=1., l=2, delta=0.02, p=2.0,eps0=2e-2,eps_d=True):
        "This part of the code is adopted from https://github.com/DimaKrotov/Biological_Learning"
        prec = 1e-50
        if eps_d:
            eps = eps0*(1-epoch/epochs)**(1.5)
        else:
            eps = eps0
        inputs = cp.transpose(batch_data)
        sig = cp.sign(self.W.T)
        tot_input = cp.dot(sig*cp.absolute(self.W.T)**(p-1),inputs)
        
        y = cp.argsort(tot_input,axis=0)
        yl = cp.zeros((self.M,self.batch_size))
        yl[y[int(self.M-1),:],cp.arange(self.batch_size)] = 1.0
        yl[y[int(self.M-l)],cp.arange(self.batch_size)] = -delta
        xx = cp.sum(cp.multiply(yl,tot_input),1)
        ds = cp.dot(yl,cp.transpose(inputs)*(R**p)) - cp.multiply(cp.tile(xx.reshape(xx.shape[0],1),(1,self.N)),self.W.T)
        nc = cp.amax(cp.absolute(ds))
        if nc < prec:
            nc = prec
        delta_W = eps*cp.transpose(cp.true_divide(ds,nc))
        self.W += delta_W

    def KH_hidden_update(self,batch_data,epoch,epochs,R=1., l=2, delta=0.02, p=2.0,eps0=2e-2,eps_d=True):
        "This part of the code is adopted from https://github.com/DimaKrotov/Biological_Learning"
        prec = 1e-50
        if eps_d:
            eps = eps0*(1-epoch/epochs)**(1.5)
        else:
            eps = eps0
        inputs = cp.transpose(self.sample_h_given_v(batch_data))
        sig = cp.sign(self.W)
        tot_input = cp.dot(sig*cp.absolute(self.W)**(p-1),inputs)
        
        y = cp.argsort(tot_input,axis=0)
        yl = cp.zeros((self.N,self.batch_size))
        yl[y[int(self.N-1),:],cp.arange(self.batch_size)] = 1.0
        yl[y[int(self.N-l)],cp.arange(self.batch_size)] = -delta
        xx = cp.sum(cp.multiply(yl,tot_input),1)
        ds = cp.dot(yl,cp.transpose(inputs)*(R**p)) - cp.multiply(cp.tile(xx.reshape(xx.shape[0],1),(1,self.M)),self.W)
        nc = cp.amax(cp.absolute(ds))
        if nc < prec:
            nc = prec
        delta_W = eps* cp.true_divide(ds,nc)
        self.W += delta_W
    
    def validation(self,data_valid):
        batch_valid_size = 1000
        n_samples = data_valid.shape[0]
        n_batches = data_valid.shape[0]//batch_valid_size
        val_mse = 0; val_pnl = 0; val_ce = 0; val_ssim = 0
        for i in range(0,n_samples,batch_valid_size):
            batch = data_valid[i:i+batch_valid_size]
            val_mse += self.reconst_error(batch)
            val_pnl += self.pseudo_neg_log_likelihood(batch)
            val_ce += self.cross_entropy(batch)
            val_ssim += self.ssim_m(batch,self.reconstruct(batch))

        val_mse /= n_batches
        val_pnl /= n_batches
        val_ce /= n_batches
        val_ssim /= n_batches
        
        self.v_mse.append(val_mse)
        self.v_pnl.append(val_pnl)
        self.v_ce.append(val_ce)
        self.v_ssim.append(val_ssim)
        
        return val_mse, val_pnl, val_ce, val_ssim
    
    def load_parameters(self, name):
        loaded_parameters = np.load(name+'_parameters.npz')

        # Assign the loaded parameters to the RBM variables
        self.W = loaded_parameters['weights']
        self.W = cp.array(self.W)
        self.a = loaded_parameters['v_biases']
        self.a = cp.array(self.a)
        self.b = loaded_parameters['h_biases']
        self.b = cp.array(self.b)
    
    def train(self,data_train,data_valid,training_settings,incr=10,save_checkpoints=False,track_learning=False,
              save_learn_funcs=False,save_params=False,plot_weights=False):
        training = training_settings['training']
        epochs = training_settings['epochs']
        k = training_settings['k']
        KH = training_settings['KH']
        R = training_settings['R']
        l = training_settings['l']
        p = training_settings['p']
        delta = training_settings['delta']
        eps0 = training_settings['eps0']
        eps_d = training_settings['eps_d']
        dataset = training_settings['dataset']
        seed = training_settings['seed']
        label = training_settings['label']
        addrss = training_settings['addrss']
        
        n_minibatches = data_train.shape[0] // self.batch_size
        
        eps_name = '_eps0' if not eps_d else '_epsD'  
        KH_name = '' if not KH else '_'+KH+'_l'+str(l)+'_delta'+str(delta)+'_p'+str(p)+eps_name+str(eps0)+'_'+'R'+str(R)
        rbm_name = training+'k'+str(k)+'_M'+str(self.M)+'_eta'+str(self.eta)+'_'+self.W_init+'_B'+str(self.batch_size)
        det_name = label+dataset+'_seed'+str(seed)+'_epochs'+str(epochs)+'_RBM_'
        name = det_name+rbm_name+KH_name
        
        checkpoints = [50,100,200,350]
        c_ind = 0
        for epoch in range(epochs):
            data_train = data_train[cp.random.permutation(data_train.shape[0]),:]
            tr_mse = 0; tr_pnl = 0; tr_ce = 0; vl_mse = 0; vl_pnl = 0; vl_ce = 0; vl_ssim = 0
            for x in range(n_minibatches):
                batch_data = data_train[x*self.batch_size:(x+1)*self.batch_size,:] 
                if KH == 'bottom-up':
                    self.KH_update(batch_data,epoch,epochs,R,l,delta,p,eps0,eps_d)
                if KH == 'top-down':
                    self.KH_hidden_update(batch_data,epoch,epochs,R,l,delta,p,eps0,eps_d)
                else:
                    pass
                
                self.minibatching_gradient_descent(batch_data, k, training, epoch, epochs)
                    
                if track_learning:
                    tr_mse += self.reconst_error(batch_data)
                    tr_pnl += self.pseudo_neg_log_likelihood(batch_data)
                    tr_ce += self.cross_entropy(batch_data)
            tr_mse /= n_minibatches
            tr_pnl /= n_minibatches
            tr_ce /= n_minibatches
            
            if save_checkpoints:
                if epoch == checkpoints[c_ind]:
                    c_name = '_cpoint'+str(checkpoints[c_ind]+1)
                    try:
                        if cp==cupy:
                            weights_ = cp.asnumpy(self.W)
                            v_biases_ = cp.asnumpy(self.a)
                            h_biases_ = cp.asnumpy(self.b)
                    except:
                        weights_ = self.W
                        v_biases_ = self.a
                        h_biases_ = self.b
                    
                    cp.savez(name+c_name+'_parameters.npz', weights=weights_, v_biases=v_biases_, h_biases=h_biases_)
                    c_ind += 1
            
            if track_learning:
                self.t_mse.append(tr_mse)
                self.t_pnl.append(tr_pnl)
                self.t_ce.append(tr_ce)
                vl_mse,vl_pnl,vl_ce,vl_ssim = self.validation(data_valid)
                if incr !=0 :
                    if (epoch + 1) % incr == 0:
                        print(f"Epoch {epoch + 1}/{epochs}, Training Data Reconstructions:")
                        self.plot_samples(batch_data) 
                        print(f"Epoch {epoch + 1}/{epochs}, Training MSE: {self.t_mse[epoch]:.7f}")
                        print(f"Epoch {epoch + 1}/{epochs}, Validation MSE: {self.v_mse[epoch]:.7f}")
                        print(f"Epoch {epoch + 1}/{epochs}, Validation Data Reconstructions:")
                        self.plot_samples(data_valid)
                        if plot_weights:
                            print('Receptive Fields:')
                            self.plot_weights_hid()
                    
        if save_learn_funcs:
            learn_funcs = [cp.array(self.t_pnl),cp.array(self.t_ce),cp.array(self.t_mse),cp.array(self.v_pnl),cp.array(self.v_ce),
                          cp.array(self.v_mse),cp.array(self.v_ssim)]
            learn_funcs_st = ['_tr_pnl.npy','_tr_ce.npy','_tr_mse.npy','_val_pnl.npy','_val_ce.npy','_val_mse.npy','_val_ssim.npy']
            for i in range(len(learn_funcs_st)):
                try:
                    if cp==cupy:
                        func_ = cp.asnumpy(learn_funcs[i])
                except:
                    func_ = learn_funcs[i]
                cp.save(addrss+name+learn_funcs_st[i],func_)
        if save_params:
            try:
                if cp==cupy:
                    weights_ = cp.asnumpy(self.W)
                    v_biases_ = cp.asnumpy(self.a)
                    h_biases_ = cp.asnumpy(self.b)
            except:
                weights_ = self.W
                v_biases_ = self.a
                h_biases_ = self.b
            cp.savez(addrss+name+'_parameters.npz', weights=weights_, v_biases=v_biases_, h_biases=h_biases_)
