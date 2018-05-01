import numpy as np
import mne

from scipy.io  import loadmat
import conversion  

def _import_data(path,subject=1,session=1,task=1):
        
    # include a / at the end if is not there already      
    if not(path.split('/')[-1] == ''):
        path = path + '/'
        
    if type(session) is list:
        
        raw_list = []
        for s in session:
            raw,event_id = _import_data(path,subject,s,task)
            raw_list.append(raw)
        return mne.io.concatenate_raws(raw_list), event_id    

    else:

        if path.endswith('motorimagery/BCI-competitions/BCI-IV/2a/'):
            return get_bci_iv_2a(path,subject,session,task)       
            
        elif path.endswith('motorimagery/BCI-competitions/BCI-III/IIIa/'):
            return get_bci_iii_3a(path,subject,session,task)  
    
        elif path.endswith('motorimagery/BCI-competitions/BCI-III/IVa/'):
            return get_bci_iii_4a(path,subject,session,task)               
            
        elif path.endswith('motorimagery/BNCI/2-two-class-motor-imagery/'):
            return get_bnci_two_class_motor_imagery(path,subject,session,task)                 
    
        elif path.endswith('motorimagery/BNCI/13-individual-imagery/'):
            return get_bnci_individual_imagery(path,subject,session,task) 
    
        elif path.endswith('motorimagery/Physionet/eegmmidb/'):
            return get_physionet_motormovementimagery(path,subject,
                                                      session,task) 
            
        elif path.endswith('erp/BNCI/8-P300-speller-ALS-patients/'):
            return get_bnci_erp_als(path,subject,session,task)      
            
        elif path.endswith('erp/BCI-competitions/BCI-III/II/'):
            return get_bci_iii_p300(path,subject,session,task)

        elif path.endswith('erp/BrainInvaders/'):
            return get_brain_invaders_p300(path,subject,session,task)
    
        else:
            print 'Please, select a valid dataset path'
            return None            
    
def get_bci_iv_2a(path,subject=1,session=1,task=1):    
    
    if task > 1:
        print "This dataset has just one task!"
        return None
    if session > 2:
        print "This dataset has just two sessions per subject!" 
        return None                   

    description = 'BCI IV Competition - Dataset 2a'  

    ch_names = []

    mat = loadmat(path + 'chnames.mat')    
    nchannels = mat['chnames'].shape[1]
    ch_names = []
    for n in range(nchannels):                    
        ch_names.append(mat['chnames'][0][n][0])
    ch_names = ch_names + ['eog' + str(i+1) for i in range(3)]    
    ch_names += ['stim']             
                                                          
    ch_types = []
    ch_types = ch_types + ['eeg']*22
    ch_types = ch_types + ['eog']*3
    ch_types = ch_types + ['stim']

    event_id  = dict(left=1,right=2,feet=3,tongue=4)
                  
    suffix   = ['T','E'][session-1]
    filename = 'A0' + str(subject) + suffix + '.mat'
    filepath = path + filename
    mat      = loadmat(filepath)
    fs       = mat['fs']
    signal   = mat['signal'].T.astype('float') # nchannels, nsamples  
    raw      = conversion.create_mne_raw(signal,fs,ch_names,
                                         ch_types,description)  
                
    return raw, event_id
        
def get_bci_iii_3a(path,subject=1,session=1,task=1):    
    
    if task > 1:
        print "This dataset has just one task!"
        return None
    if session > 1:
        print "This dataset has only one session per subject!" 
        return None                   

    description = 'BCI III Competition - Dataset IIIa'    

    ch_names = []                        
    ch_names = ch_names + ['ch' + str(i+1) for i in range(60)]
    ch_names = ch_names + ['stim']    
                                                          
    ch_types = []
    ch_types = ch_types + ['eeg']*60
    ch_types = ch_types + ['stim']

    event_id  = dict(left=1,right=2,feet=3,tongue=4)
                  
    filename = 'subject' + str(subject) + '.mat'
    filepath = path + filename
    mat      = loadmat(filepath)
    fs       = mat['fs']
    signal   = mat['signal'].T.astype('float') # nchannels, nsamples  
    raw      = conversion.create_mne_raw(signal,fs,ch_names,
                                         ch_types,description)  
                
    return raw, event_id    
    
def get_bci_iii_4a(path,subject=1,session=1,task=1):    
       
    if task > 1:
        print "This dataset has just one task!"
        return None
    if session > 1:
        print "This dataset has only one session per subject!" 
        return None                   

    description = 'BCI III Competition - Dataset IVa'    

    ch_names = []

    mat = loadmat(path + 'chnames.mat')    
    nchannels = mat['chnames'].shape[1]
    ch_names = []
    for n in range(nchannels):                    
        ch_names.append(mat['chnames'][0][n][0])
    ch_names += ['stim']    
                                                          
    ch_types = []
    ch_types = ch_types + ['eeg']*nchannels
    ch_types = ch_types + ['stim']

    event_id  = dict(hand=1,feet=2)
                  
    filename = 'subject' + str(subject) + '.mat'
    filepath = path + filename
    mat      = loadmat(filepath)
    fs       = mat['fs']
    signal   = mat['signal'].T.astype('float') # nchannels, nsamples  
    raw      = conversion.create_mne_raw(signal,fs,ch_names,
                                         ch_types,description)  
                
    return raw, event_id      
               
def get_bnci_two_class_motor_imagery(path,subject=1,session=1,task=1):   
    
    if task > 1:
        print "This dataset has only one task!"   
        return None
    if session > 1:
        print "This dataset has only one session per subject!"              
        return None    

    description = 'Two class motor imagery database (from BNCI)'   

    ch_names = []                        
    ch_names = ch_names + ['ch' + str(i+1) for i in range(15)]
    ch_names = ch_names + ['stim']    
                                                          
    ch_types = []
    ch_types = ch_types + ['eeg']*15
    ch_types = ch_types + ['stim'] 

    event_id  = dict(right=1,feet=2)
               
    # getting the training data for this subject 
    if subject < 10:
        filename = 'S0' + str(subject) + 'T.mat'
    else:
        filename = 'S'  + str(subject)  + 'T.mat'    
    filepath = path + filename
    mat      = loadmat(filepath)
    fs       = mat['fs']
    signal   = mat['signal'].T.astype('float') # nchannels, nsamples    
    raw      = conversion.create_mne_raw(signal,fs,ch_names,
                                         ch_types,description)  
                      
    return raw, event_id        
    
def get_bnci_individual_imagery(path,subject=1,session=1,task=1):

    if task > 1:
        print "This dataset has only one task!"   
        return None
    if session > 1:
        print "This dataset has just two sessions per subject!"              
        return None   
    
    description = 'Individual motor imagery database (from BNCI)'
    
    ch_names = []  
    mat = loadmat(path + 'chnames.mat')    
    nchannels = mat['chnames'].shape[1]
    ch_names = []
    for n in range(nchannels):                    
        ch_names.append(mat['chnames'][0][n][0])
    ch_names += ['stim']    
                                                          
    ch_types = []
    ch_types = ch_types + ['eeg']*nchannels
    ch_types = ch_types + ['stim'] 

    event_id = dict(word=1,sub=2,nav=3,hand=4,feet=5)
    subjects = ['A','C','D','E','F','G','H','J','L']
                          
    filename = str(subjects[subject]) + str(session) + '.mat'
    filepath = path + filename
    mat      = loadmat(filepath)
    fs       = mat['fs']
    signal   = mat['signal'].T.astype('float') # nchannels, nsamples    
    raw      = conversion.create_mne_raw(signal,fs,ch_names,
                                         ch_types,description)                    

    return raw, event_id    
    
def get_physionet_motormovementimagery(path,subject=1,session=1,task=1):
        
    if task > 6:
        print "This dataset has only six tasks!"   
        return None
    if session > 1:
        print "This dataset has only one session per subject!"              
        return None     
    
    description = 'EEG Motor Movement/Imagery Dataset'
    
    if subject < 10:
        subject_preffix = 'S00' + str(subject)    
    elif subject >= 10 and subject < 100:
        subject_preffix = 'S0'  + str(subject)
    else:
        subject_preffix = 'S'   + str(subject)
        
    task_runs = [[3,7,11],[4,8,12],[5,9,13],[6,10,14],[1],[2]]  
    raw_run   = []    
    for run in task_runs[task-1]:        
        if run < 10:
            run_suffix = '0' + str(run)
        else:
            run_suffix = str(run)        
        file_path = path + subject_preffix + '/' 
        file_path += subject_preffix + 'R' + run_suffix + '.edf'   
        raw_run.append(mne.io.read_raw_edf(file_path, preload=True, verbose=False))

    raw = mne.io.concatenate_raws(raw_run)               
    raw.info['description'] = description
    raw.rename_channels(lambda x: x.strip('.'))

    if task in [1,2]:
        event_id = dict(left=2,right=3)
    if task in [3,4]:
        event_id = dict(hands=2,feet=3)
    if task in [5,6]:
        event_id = {}

    return raw, event_id  
    
def get_bnci_erp_als(path,subject=1,session=1,task=1):

    if task > 1:
        print "This dataset has just one task!"
        return None
    if session > 1:
        print "This dataset has just one session per subject!" 
        return None                   

    description = 'P300 speller with ALS patients (from BNCI)'  

    ch_names = []

    mat = loadmat(path + 'chnames.mat')    
    nchannels = mat['chnames'].shape[1]
    ch_names = []
    for n in range(nchannels):                    
        ch_names.append(mat['chnames'][0][n][0])
    ch_names += ['stim']             
                                                          
    ch_types = []
    ch_types = ch_types + ['eeg']*nchannels
    ch_types = ch_types + ['stim']

    event_id  = dict(target=2,nontarget=1)
                  
    filename = 'A0' + str(subject) + '.mat'
    filepath = path + filename
    mat      = loadmat(filepath)
    fs       = mat['fs']
    signal   = mat['signal'].T.astype('float') # nchannels, nsamples  
    raw      = conversion.create_mne_raw(signal,fs,ch_names,
                                         ch_types,description) 
    
    return raw, event_id
    
def get_bci_iii_p300(path,subject=1,session=1,task=1):    
    
    if task > 1:
        print "This dataset has just one task!"
        return None
    if session > 2:
        print "This dataset has just two sessions per subject!" 
        return None    
        
    description = 'BCI III Competition - Dataset II (P300)'         
        
    ch_names = []                        
    ch_names = ch_names + ['ch' + str(i+1) for i in range(64)]
    ch_names = ch_names + ['stim']          
        
    event_id  = dict(target=2,nontarget=1)                        
    ch_types = []
    ch_types = ch_types + ['eeg']*64
    ch_types = ch_types + ['stim']    
    
    filename = 'Subject' + str(subject) + '_part' + str(session) + '.mat'
    filepath = path + filename
    mat      = loadmat(filepath)
    fs       = mat['fs']
    signal   = mat['signal'].T.astype('float') # nchannels, nsamples  
    raw      = conversion.create_mne_raw(signal,fs,ch_names,
                                         ch_types,description)                         
    
    return raw, event_id    

def get_brain_invaders_p300(path,subject=1,session=1,task=1):    
    
    if task > 1:
        print "This dataset has just one task!"
        return None
    if session > 1:
        print "This dataset has just one session per subject!" 
        return None    
        
    description = 'P300 from Brain Invaders'         
        
    ch_names = []                        
    ch_names = ch_names + ['ch' + str(i+1) for i in range(32)]
    ch_names = ch_names + ['stim']          
        
    event_id  = dict(target=2,nontarget=1)                        
    ch_types = []
    ch_types = ch_types + ['eeg']*32
    ch_types = ch_types + ['stim']    
    
    filename = 'subject' + "{0:0>2}".format(subject) + '.mat'
    filepath = path + filename
    mat      = loadmat(filepath)
    fs       = mat['fs']
    signal   = mat['signal'].T.astype('float') # nchannels, nsamples  
    raw      = conversion.create_mne_raw(signal,fs,ch_names,
                                         ch_types,description)                         
    
    return raw, event_id         

    
    
    
    
    
        
        
        
        
        
        