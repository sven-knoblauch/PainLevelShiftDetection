
import pandas as pd
import neurokit2 as nk
import numpy as np
from scipy import stats
from scipy.io import loadmat




class DatasetTimeFrameGeneratorINTENSE:
    
    def __init__(self, path, subject, saving_folder=".\\", drug_delta=5, timestamps=None):
        self.path = path
        self.subject = subject
        self.drug_delta = drug_delta
        self.saving_folder = saving_folder
        self.timestamps = timestamps
        self.fs = 512
        self.load_data()

    #load data from mat files
    def load_data(self):
        self.data = loadmat(self.path+self.subject+".mat")
        self.biodata = {"cor": self.data["CORbt"].flatten(),
                        "zyg": self.data["ZYGbt"].flatten(),
                        "eda": self.data["SCL_filt"].flatten(),
                        "ecg": self.data["ECG"].flatten()}
        if self.timestamps is None:
            self.timestamps = self.data["morphin_timestamps"].flatten()
        del self.data

    #generate features with the timestamps of the drug inducing
    def generate_features(self):
        starts = self.timestamps-10*self.fs-5*60*self.fs
        ends = self.timestamps+5*60*self.fs+10*self.fs

        ecg_features = pd.DataFrame([])
        eda_features = pd.DataFrame([])
        cor_features = pd.DataFrame([])
        zyg_features = pd.DataFrame([])

        for a, b in list(zip(starts, ends)):
            try:
                ecg_f = self.generate_ecg_features(self.biodata["ecg"][a:b])
                eda_f = self.generate_eda_features(self.biodata["eda"][a:b])
                zyg_f = self.generate_emg_features(self.biodata["zyg"][a:b], "zyg")
                cor_f = self.generate_emg_features(self.biodata["cor"][a:b], "cor")

                ecg_features = pd.concat([ecg_features, ecg_f], ignore_index=True)
                eda_features = pd.concat([eda_features, eda_f], ignore_index=True)
                zyg_features = pd.concat([zyg_features, zyg_f], ignore_index=True)
                cor_features = pd.concat([cor_features, cor_f], ignore_index=True)
            except:
                continue
            
        ecg_features["subject"] = self.subject.split("_")[0]
        ecg_features["pain"] = (ecg_features["Event_Onset"]<320).astype(int)
        ecg_features = ecg_features.drop(["Event_Onset"], axis=1)
        return pd.concat([ecg_features.reset_index(drop=True), eda_features.reset_index(drop=True), zyg_features.reset_index(drop=True), cor_features.reset_index(drop=True)], axis=1)

    #generate emg feature from emg signal
    def generate_emg_features(self, emg, name="emg_"):
        #prepare data
        processed = nk.emg_process(emg, sampling_rate=self.fs)
        epochs = nk.epochs_create(data=processed, events = [5, 10+5*60, 10+10*60], sampling_rate = self.fs, epochs_start = 0, epochs_end = 5)

        emg_features_clean = self.calculate_statistical_features(epochs, name=name+"_clean_", key="EMG_Clean")
        emg_features_amplitude = self.calculate_statistical_features(epochs, name=name+"_amplitude_", key="EMG_Amplitude")
        emg_frequ_features = self.generate_frequence_features(epochs, name=name+"_", key="EMG_Clean")
        return pd.concat([emg_features_clean.reset_index(drop=True), emg_features_amplitude.reset_index(drop=True), emg_frequ_features.reset_index(drop=True)], axis=1)

    #generate eda features from an eda signal
    def generate_eda_features(self, eda):
        #prepare data
        processed = nk.eda_process(eda, sampling_rate=self.fs)
        epochs = nk.epochs_create(data=processed, events = [5, 10+5*60, 10+10*60], sampling_rate = self.fs, epochs_start = 0, epochs_end = 5)
        #geenrate features of tonic, phasic and clean signal
        eda_features_tonic = self.calculate_statistical_features(epochs, name="eda_tonic_", key="EDA_Tonic")
        eda_features_phasic = self.calculate_statistical_features(epochs, name="eda_phasic_", key="EDA_Phasic")
        eda_features_clean = self.calculate_statistical_features(epochs, name="eda_clean_", key="EDA_Clean")
        generated_feature = pd.concat([eda_features_tonic.reset_index(drop=True), eda_features_phasic.reset_index(drop=True), eda_features_clean.reset_index(drop=True)], axis=1)
        analyzed_eda = nk.eda_analyze(epochs)
        analyzed_eda = analyzed_eda[["EDA_Peak_Amplitude"]]
        return pd.concat([analyzed_eda.reset_index(drop=True), generated_feature.reset_index(drop=True)], axis=1)

    #geenrate ecg features of ecg signal
    def generate_ecg_features(self, ecg):
        #prepare data
        processed = nk.ecg_process(ecg, sampling_rate=self.fs)
        epochs = nk.epochs_create(data=processed, events = [5, 10+5*60, 10+10*60], sampling_rate = self.fs, epochs_start = 0, epochs_end = 5)
        
        #calculate features
        ecg_features_hr = self.calculate_statistical_features(epochs, name="ecg_hr_", key="ECG_Rate")
        ecg_features_clean = self.calculate_statistical_features(epochs, name="ecg_clean_", key="ECG_Clean")
        analyzed_ecg = nk.ecg_analyze(epochs)
        analyzed_ecg = analyzed_ecg.drop(["Label", "ECG_Phase_Atrial", "ECG_Phase_Ventricular", "ECG_Quality_Mean"], axis=1)
        return pd.concat([analyzed_ecg.reset_index(drop=True), ecg_features_hr.reset_index(drop=True), ecg_features_clean.reset_index(drop=True)], axis=1)



    #calculate statistical features
    def calculate_statistical_features(self, epochs, name="signal_", key="Signal"):
        tmp = []
        for epoch_key in epochs.keys():
            signal = np.array(epochs[epoch_key][key])
            singal_max = np.max(signal)
            singal_min = np.min(signal)
            singal_mean = np.mean(signal)
            singal_std = np.std(signal)
            singal_var = np.var(signal)
            singal_rms = np.sqrt(np.mean(signal**2))
            singal_power = np.mean(signal**2)
            singal_peak = np.max(np.abs(signal))
            singal_p2p = np.ptp(signal)
            singal_skew = stats.skew(signal)
            singal_kurtosis = stats.kurtosis(signal)
            singal_crestfactor = np.max(np.abs(signal))/np.sqrt(np.mean(signal**2))
            singal_formfactor = np.sqrt(np.mean(signal**2))/np.mean(signal)
            singal_pulseindicator = np.max(np.abs(signal))/np.mean(signal)
            
            #from paper of tobias
            chunks = list(map(np.std, np.array_split(signal, len(signal)//4)))
            singal_var_second_moment = np.mean(np.array(chunks))
            singal_variation_second_moment = np.mean((chunks-singal_var_second_moment)**2)
            singal_std_second_moment = np.sqrt(singal_variation_second_moment)
            singal_mean_val_first_diff = np.mean(signal[1:]-signal[:-1])
            singal_mean_abs_val_first_diff = np.mean(np.abs(signal[1:]-signal[:-1]))
            singal_mean_abs_val_second_diff = np.mean(np.abs(signal[2:]-signal[:-2]))

            tmp.append([singal_max, singal_min, singal_mean, singal_std, singal_var, singal_rms, singal_power, singal_peak, singal_p2p, singal_skew,
                        singal_kurtosis, singal_crestfactor, singal_formfactor, singal_pulseindicator, singal_var_second_moment, singal_variation_second_moment,
                        singal_std_second_moment, singal_mean_val_first_diff, singal_mean_abs_val_first_diff, singal_mean_abs_val_second_diff])
            col = [name+"max", name+"min", name+"mean", name+"std", name+"var", name+"rms", name+"power",
                                name+"peak", name+"p2p", name+"skew", name+"kurtosis", name+"crestfactor", name+"formfactor",
                                name+"pulseindicator", name+"var_second_moment", name+"variation_second_moment", name+"std_second_moment",
                                name+"mean_val_first_diff", name+"mean_abs_val_first_diff", name+"mean_abs_val_second_diff"]

        return pd.DataFrame(data=tmp, columns=col)


    #generate statistical frequence features
    def generate_frequence_features(self, epochs, name="signal_", key="Signal"):
        tmp = []
        for epoch_key in epochs.keys():
            signal = np.array(epochs[epoch_key][key])
            signal_fft = np.fft.fft(signal)
            s = np.abs(signal_fft**2)/len(signal_fft)
            singal_max = np.max(s)
            singal_sum = np.sum(s)
            singal_mean = np.mean(s)
            signal_var = np.var(s)
            signal_peak = np.max(np.abs(s))
            signal_skew = stats.skew(s)
            signal_kurtosis = stats.kurtosis(s)
            tmp.append([singal_max, singal_sum, singal_mean, signal_var, signal_peak, signal_skew, signal_kurtosis])
        col = [name+"fft_max", name+"fft_sum", name+"fft_mean", name+"fft_var", name+"fft_peak", name+"fft_skew", name+"fft_kurtosis"]
        return pd.DataFrame(data = tmp, columns=col)

    #save features in pickle file
    def save_features(self, features):
        features.to_pickle(self.saving_folder+self.subject.split("_")[0]+".pkl")