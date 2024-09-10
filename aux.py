##################################
## Auxiliary Functions
##################################
def binarize_data(X, th=127):
    # th: threshold
    X[X < th] = 0
    X[X >= th] = 1
    return X 

def get_dataset(dataset='mnist'):
    # 'kmnist', 'mnist'
    import tensorflow_datasets as tfds
    from tensorflow.keras.utils import to_categorical
    import numpy as np

    np.random.seed(42)

    # Load the dataset
    kmnist_train, kmnist_test = tfds.load(dataset, split=['train', 'test'], as_supervised=True)

    # Shuffle and preprocess the training data
    kmnist_train = kmnist_train.shuffle(buffer_size=10000, seed = 42)

    # Extract data and labels for training, validation, and test sets
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for image, label in tfds.as_numpy(kmnist_train):
        train_data.append(image.flatten())
        train_labels.append(label)

    for image, label in tfds.as_numpy(kmnist_test):
        test_data.append(image.flatten())
        test_labels.append(label)

    # Convert lists to numpy arrays
    train_data = np.array(train_data).astype(np.float32)
    train_labels = np.array(train_labels)

    indices = np.arange(len(train_data))
    np.random.shuffle(indices)

    # Use the shuffled indices to split the dataset
    split_index = 50000
    train_indices, val_indices = indices[:split_index], indices[split_index:]

    # Split the dataset
    train_data, valid_data = train_data[train_indices], train_data[val_indices]
    train_labels, valid_labels = train_labels[train_indices], train_labels[val_indices]

    test_data = np.array(test_data).astype(np.float32)
    test_labels = np.array(test_labels)

    train_labels = to_categorical(train_labels)
    valid_labels = to_categorical(valid_labels)
    test_labels = to_categorical(test_labels)

    print("Training Images Shape:", train_data.shape)
    print("Training Labels Shape:", train_labels.shape)
    print("Validation Images Shape:", valid_data.shape)
    print("Validation Labels Shape:", valid_labels.shape)
    print("Test Images Shape:", test_data.shape)
    print("Test Labels Shape:", test_labels.shape)
    return train_data, valid_data, test_data, train_labels, valid_labels, test_labels

def load_files(d_name='_tr',training='CD',k=1,M=100,eta=0.1,KH='hKH',l=2,delta=0.4,p=2.0,eps0=2e-3,label='',addrss='out/',
               epochs=500,batch_size=100,R=1.0,W_init='std',eps_d=True,dataset='mnist',seed=1234):
    import numpy as np
    
    eps_name = '_eps0' if not eps_d else '_epsD'  
    KH_name = '' if not KH else '_'+KH+'_l'+str(l)+'_delta'+str(delta)+'_p'+str(p)+eps_name+str(eps0)+'_'+'R'+str(R)
    rbm_name = training+'k'+str(k)+'_M'+str(M)+'_eta'+str(eta)+'_'+W_init+'_B'+str(batch_size)
    det_name = label+dataset+'_seed'+str(seed)+'_epochs'+str(epochs)+'_RBM_'
    name = det_name+rbm_name+KH_name


    mse = np.load(addrss+name+d_name+'_mse.npy')
    pnl = np.load(addrss+name+d_name+'_pnl.npy')
    ce = np.load(addrss+name+d_name+'_ce.npy')
    return np.vstack((pnl,mse,ce))

def load_acc(training='CD',k=1,M=100,eta=0.1,KH='hKH',l=2,delta=0.4,p=2.0,eps0=2e-3,label='',addrss='out/',
               epochs=500,batch_size=100,R=1.0,W_init='std',eps_d=True,dataset='mnist',seed=1234):
    import numpy as np
    
    eps_name = '_eps0' if not eps_d else '_epsD'  
    KH_name = '' if not KH else '_'+KH+'_l'+str(l)+'_delta'+str(delta)+'_p'+str(p)+eps_name+str(eps0)+'_'+'R'+str(R)
    rbm_name = training+'k'+str(k)+'_M'+str(M)+'_eta'+str(eta)+'_'+W_init+'_B'+str(batch_size)
    det_name = label+dataset+'_seed'+str(seed)+'_epochs'+str(epochs)+'_classRBM_'
    name = det_name+rbm_name+KH_name

    v_acc = np.load(addrss+name+'_val_acc.npy')
    return v_acc

def plot_funcs(data,inp_label,data2,limits,sc='l',inset=True,save=False,name=''):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    func_label= [r'$\tilde{\mathcal{L}}$','Reconstruction MSE',r'$\mathbb{H}$']
    
    col = ['#000000','#e6194b', '#3cb44b', '#4363d8', '#f58231', '#B8860B', '#f032e6', '#800000','#fabebe', '#bcf60c','#008080', '#e6beff', 
           '#9a6324', '#fffac8', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080','#911eb4','#ffe119','#46f0f0']
    
    label_iterator = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
    ax_labs = ['Training','Validation']
    intrvl = 25
    
    def funcs(data):
        func1 = []
        func2 = []
        func3 = []
        for i in range(len(data)):
            func1.append(data[i][0,:])
            func2.append(data[i][1,:])
            func3.append(data[i][2,:])
        return [func1,func2,func3]
    
    if not data2:
        cm = plt.get_cmap('gist_rainbow')
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        all_funcs = funcs(data)
        dummy = 0
        
        for i in range(len(all_funcs)):
            for j in range(len(inp_label)):
                axs[i].plot(all_funcs[i][j],c=col[j],label=inp_label[j])
                axs[i].set_xlabel('Epochs',fontsize=16)
                axs[i].set_ylabel(func_label[i],fontsize=16)
                axs[i].text(0.07, 0.95, label_iterator[i], transform=axs[i].transAxes,
                               va='top', ha='left', fontsize=12)
                axs[i].set_facecolor("lightgrey")
                if i==0:
                    axs[i].legend(loc='best')
                dummy+=1
        if inset:
            for i in range(len(all_funcs)):
                ax = axs[i]
                # inset axes....
                x1, x2, y1, y2 = limits[i]
                axins = ax.inset_axes(
                    [0.2, 0.23, 0.75, 0.65],
                    xlim=(x1, x2), ylim=(y1, y2))
            for j in range(len(inp_label)):
                axins.plot(all_funcs[i][j], c=col[j], label=inp_label[j])
                axins.set_facecolor("lightgrey")
                ax.indicate_inset_zoom(axins)
        plt.tight_layout()
        plt.show()
    else:
        cm = plt.get_cmap('gist_rainbow')
        fig, axs = plt.subplots(2, 3, figsize=(14, 8))
        all_funcs = funcs(data)
        all_funcs2 = funcs(data2)
        font_size = 18
        
        if sc=='xl':
            axlims = [0.3, 0.3, 0.65, 0.65]
        if sc=='l':
            axlims = [0.4, 0.4, 0.55, 0.55]
        if sc=='m':
            axlims = [0.45, 0.45, 0.5, 0.5]
        if sc=='s':
            axlims = [0.48, 0.48, 0.5, 0.5]
        
        for i in range(len(all_funcs)):
            for j in range(len(inp_label)):
                ax = axs[0, i]
                ax.plot(all_funcs[i][j], c=col[j], label=inp_label[j])
                ax.set_ylabel(func_label[i], fontsize=font_size)
                ax.text(0.07, 0.95, label_iterator[i], transform=ax.transAxes, va='top', ha='left', fontsize=font_size)
                ax.set_facecolor("whitesmoke")

        if inset:
            for i in range(len(all_funcs)):
                ax = axs[0, i]
                # inset axes....
                #x1, x2, y1, y2 = 50, 15001, 0.068, 0.08  # subregion of the original image
                x1, x2, y1, y2 = limits[0][i]
                axins = ax.inset_axes(
                        axlims,
                        xlim=(x1, x2), ylim=(y1, y2))
                for j in range(len(inp_label)):
                    axins.plot(all_funcs[i][j], c=col[j], label=inp_label[j])
                    axins.set_facecolor("whitesmoke")
                    ax.indicate_inset_zoom(axins)

        for i in range(len(all_funcs2)):
            for j in range(len(inp_label)):
                ax = axs[1, i]
                ax.plot(all_funcs2[i][j], c=col[j], label=inp_label[j])
                ax.set_xlabel('Epochs', fontsize=font_size)
                ax.set_ylabel(func_label[i], fontsize=font_size)
                ax.text(0.07, 0.95, label_iterator[i+3], transform=ax.transAxes, va='top', ha='left', fontsize=font_size)
                ax.set_facecolor("whitesmoke")
        if inset:
            for i in range(len(all_funcs2)):
                ax = axs[1, i]
                # inset axes....
                #x1, x2, y1, y2 = (50, 15001, 0.068, 0.08)  # subregion of the original image
                x1, x2, y1, y2 = limits[1][i]
                axins = ax.inset_axes(
                    axlims,
                    xlim=(x1, x2), ylim=(y1, y2))
                for j in range(len(inp_label)):
                    axins.plot(all_funcs2[i][j], c=col[j], label=inp_label[j])
                    axins.set_facecolor("whitesmoke")
                    ax.indicate_inset_zoom(axins)
        axs[0,0].text(-0.3, 0.66, ax_labs[0],c='navy', transform=axs[0,0].transAxes,
                       va='top', ha='left', fontsize=20, rotation=90.)
        axs[1,0].text(-0.3, 0.7, ax_labs[1],c='navy', transform=axs[1,0].transAxes,
                       va='top', ha='left', fontsize=20, rotation=90.)
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.52, -0.1), fontsize=16)
        fig.tight_layout(pad=1.0)
        if save:
            plt.savefig('outs/'+name+'.pdf', bbox_inches='tight',dpi=600)
        # Show the plot
        plt.show()