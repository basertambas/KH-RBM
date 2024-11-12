import numpy as cp
try:
    import cupy
    if cupy.cuda.is_available():
        cp = cupy
except:
    pass
import matplotlib.pyplot as plt
import time

class class_RBM:
    def __init__(self, model):
        self.N = model['N']                     # number of visible units
        self.M = model['M']                     # number of hidden units
        self.eta = model['eta']                 # learning rate of the RBM
        self.n_cl = model['n_cl']               # number of classes (C)
        self.batch_size = model['batch_size']   # batch size
        self.W_init = model['W_init']           #{'lecun', 'std'} --weight initialization
        
        if self.W_init == 'std':
            self.W = cp.random.normal(0.0, 1.0, (self.N+self.n_cl, self.M))                 # std
        else:
            self.W = cp.random.randn(self.N+self.n_cl, self.M) / cp.sqrt(self.N+self.n_cl)  # lecun

        self.a = cp.zeros((self.N+self.n_cl))
        self.b = cp.zeros((self.M))

        self.persistent_hidden_states = cp.random.randint(0, 2, size=(self.batch_size,self.M))
        self.acc_valid = []

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

    def plot_samples(self, batch_data):
        size = batch_data.shape[0] if batch_data.shape[0] < self.n_cl else self.n_cl
        idx = cp.random.choice(cp.arange(batch_data.shape[0]), size=size,replace=False)
        X_batch = batch_data[idx, :]
        v_eq = self.reconstruct(X_batch)
        v_eq = v_eq[:,:-self.n_cl]
        X_batch = X_batch[:,:-self.n_cl]
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
        Plot receptive fields.
        '''
        L_h = int((self.M)**(0.5))
        L_v = int((self.N)**(0.5))
        plt.clf()
        fig, axes = plt.subplots(L_h, L_h, gridspec_kw = {'wspace':0.1, 'hspace':0.1}, figsize=(8, 8))
        #fig.suptitle(title)
        try:
            if cp==cupy:
                W = cp.asnumpy(self.W[:-self.n_cl,:])
        except:
            W = self.W[:-self.n_cl,:]
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
        input :  (B x M)
        output :  (B x N)
        '''
        prob_v = self.activation(h.dot(self.W[:-self.n_cl,:].T) + self.a[:-self.n_cl])
        v = cp.random.rand(h.shape[0], self.N) < prob_v
        x = h.dot(self.W[self.N:,:].T) + self.a[self.N:]
        prob_l = cp.true_divide(cp.exp(x), cp.expand_dims(cp.sum(cp.exp(x), axis=1),axis=1))
        v = cp.concatenate((v,cp.eye(self.n_cl)[cp.argmax(prob_l,axis=1)]),axis=1)
        return v.astype(float)

    def sample_class_given_v(self, input_data):
        '''
        input : (B x N)
        output : (B x n_cl)
        '''
        "This function is adopted from https://github.com/rangwani-harsh/pytorch-rbm-classification/blob/master/classification_rbm.py"
        input_data = input_data[:,:-self.n_cl]
        weights = self.W[:-self.n_cl,:]
        class_weights = self.W[self.N:,:]
        class_bias = self.a[-self.n_cl:]
        hidden_bias = self.b
        num_classes = self.n_cl
        num_hidden = self.M
        precomputed_factor = cp.matmul(input_data, weights) + hidden_bias
        class_probabilities = cp.zeros((input_data.shape[0], num_classes))

        for y in range(num_classes):
            prod = cp.zeros(input_data.shape[0])
            prod += class_bias[y]
            for j in range(num_hidden):
                prod += cp.log(1 + cp.exp(precomputed_factor[:,j] + class_weights[y, j]))
            class_probabilities[:, y] = prod  

        copy_probabilities = cp.zeros(class_probabilities.shape)

        for c in range(num_classes):
            for d in range(num_classes):
                copy_probabilities[:, c] += cp.exp(-1 * class_probabilities[:, c] + class_probabilities[:, d])

        copy_probabilities = 1. / copy_probabilities
        class_probabilities = copy_probabilities

        return class_probabilities

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
        ds = cp.dot(yl,cp.transpose(inputs)*(R**p)) - cp.multiply(cp.tile(xx.reshape(xx.shape[0],1),(1,(self.N+self.n_cl))),self.W.T)
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
        yl = cp.zeros(((self.N+self.n_cl),self.batch_size))
        yl[y[int((self.N+self.n_cl)-1),:],cp.arange(self.batch_size)] = 1.0
        yl[y[int((self.N+self.n_cl)-l)],cp.arange(self.batch_size)] = -delta
        xx = cp.sum(cp.multiply(yl,tot_input),1)
        ds = cp.dot(yl,cp.transpose(inputs)*(R**p)) - cp.multiply(cp.tile(xx.reshape(xx.shape[0],1),(1,self.M)),self.W)
        nc = cp.amax(cp.absolute(ds))
        if nc < prec:
            nc = prec
        delta_W = eps* cp.true_divide(ds,nc)
        self.W += delta_W

    def c_valid(self,data_valid):
        batch_valid_size = 1000
        n_samples = data_valid.shape[0]
        n_batches = data_valid.shape[0]//batch_valid_size
        val_samp = 0
        val_p = 0
        for i in range(0,n_samples,batch_valid_size):
            batch = data_valid[i:i+batch_valid_size]
            val_samp += cp.sum(cp.equal(cp.argmax(batch[:,-self.n_cl:],axis=1),cp.argmax(self.sample_class_given_v(batch),axis=1)))

        val_samp = val_samp / n_samples

        self.acc_valid.append(val_samp)
        return val_samp

    def load_parameters(self, name):
        loaded_parameters = cp.load(name+'_parameters.npz')

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
        det_name = label+dataset+'_seed'+str(seed)+'_epochs'+str(epochs)+'_classRBM_'
        name = det_name+rbm_name+KH_name

        checkpoints = [50,100,200,350]
        c_ind = 0
        
        for epoch in range(epochs):
            data_train = data_train[cp.random.permutation(data_train.shape[0]),:]
            tr_samp_acc = 0
            tr_p_acc = 0
            for x in range(n_minibatches):
                batch_data = data_train[x*self.batch_size:(x+1)*self.batch_size,:]
                if KH == 'bottom-up':
                    self.KH_update(batch_data,epoch,epochs,R,l,delta,p,eps0,eps_d)
                if KH == 'top-down':
                    self.KH_hidden_update(batch_data,epoch,epochs,R,l,delta,p,eps0,eps_d)
                else:
                    pass

                self.minibatching_gradient_descent(batch_data, k, training, epoch, epochs)

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
                    cp.savez(name+c_name+'_parameters.npz', weights=weights_, v_biases=v_biases_, h_biases=h_biases)
                    c_ind += 1

            if track_learning:
                vl_samp = self.c_valid(data_valid)
                if incr !=0 :
                    if (epoch + 1) % incr == 0:
                        print(f"Epoch {epoch + 1}/{epochs}, Training Data Reconstructions:")
                        self.plot_samples(batch_data) 
                        print(f"Epoch {epoch + 1}/{epochs}, Validation Accuracy: {self.acc_valid[epoch]:.7f}")
                        print(f"Epoch {epoch + 1}/{epochs}, Validation Data Reconstructions:")
                        self.plot_samples(data_valid)
                        if plot_weights:
                            print('Receptive Fields:')
                            self.plot_weights_hid()
                            
        if save_learn_funcs:
            v_samp_ = cp.array(self.acc_valid)
            try:
                if cp==cupy:
                    v_samp_ = cp.asnumpy(v_samp_)
            except:
                pass
            cp.save(addrss+name+'_val_acc.npy',v_samp_)
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

