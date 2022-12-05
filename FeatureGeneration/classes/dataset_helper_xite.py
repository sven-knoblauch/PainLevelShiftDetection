from scipy.io import loadmat
import pandas as pd
import neurokit2 as nk
import numpy as np
from scipy import stats



class DatasetTimeFrameGeneratorXITE:

    def __init__(self, path, subject, window_length=5, downsampling_rate=-1, saving_folder=".\\"):
        self.path = path
        self.subject = subject
        self.window_length = window_length
        self.downsampling_rate = downsampling_rate
        self.saving_folder = saving_folder
        self.load_files()
        if downsampling_rate>0:
            self.resample_data()

    #load files from paths
    def load_files(self):
        self.sampling_rate = loadmat(self.path + "\Label\\" + self.subject + ".mat")["fs"][0,0]
        self.stimulus_timestamps = pd.read_csv(self.path + "\Stimulus\\"+self.subject+".tsv", sep='\t')
        self.biodata = loadmat(self.path + "\Bio\\"+self.subject+".mat")["data_stimuli"]
        self.event_conditions = self.stimulus_timestamps["label"]
        self.biodata = {"cor": self.biodata[:, 0],
                        "zyg": self.biodata[:, 1],
                        "eda": self.biodata[:, 3],
                        "ecg": self.biodata[:, 4],
                        "stimuli": self.biodata[:, 5]}

    #process data
    def process_data(self):
        ecg = nk.ecg_process(self.biodata["ecg"], self.sampling_rate)
        eda = nk.eda_process(self.biodata["eda"], self.sampling_rate)
        zyg = nk.emg_process(self.biodata["zyg"], self.sampling_rate)
        cor = nk.emg_process(self.biodata["cor"], self.sampling_rate)
        self.biodata = {"cor": cor,
                        "zyg": zyg,
                        "eda": eda,
                        "ecg": ecg,
                        "stimuli": self.biodata["stimuli"]}

    def resample_data(self):
        ecg = nk.signal_resample(self.biodata["ecg"], sampling_rate=self.sampling_rate, desired_sampling_rate=self.downsampling_rate)
        eda = nk.signal_resample(self.biodata["eda"], sampling_rate=self.sampling_rate, desired_sampling_rate=self.downsampling_rate)
        zyg = nk.signal_resample(self.biodata["zyg"], sampling_rate=self.sampling_rate, desired_sampling_rate=self.downsampling_rate)
        cor = nk.signal_resample(self.biodata["cor"], sampling_rate=self.sampling_rate, desired_sampling_rate=self.downsampling_rate)
        stimuli = nk.signal_resample(self.biodata["stimuli"], sampling_rate=self.sampling_rate, desired_sampling_rate=self.downsampling_rate)
        self.sampling_rate = self.downsampling_rate
        self.biodata = {"cor": cor,
                        "zyg": zyg,
                        "eda": eda,
                        "ecg": ecg,
                        "stimuli": stimuli}

    #load epochs with given events
    def load_epochs(self, data, start):
        return nk.epochs_create(data = data,
                                events = self.stimulus_timestamps["time"]/1000+start*1000,
                                sampling_rate = self.sampling_rate,
                                epochs_start = 0,
                                epochs_end = self.window_length,
                                event_conditions = self.event_conditions)

    #generate epochs from processed data
    def generate_epochs(self, start):
        ecg = self.load_epochs(self.biodata["ecg"], start)
        eda = self.load_epochs(self.biodata["eda"], start)
        cor = self.load_epochs(self.biodata["cor"], start)
        zyg = self.load_epochs(self.biodata["zyg"], start)
        stimuli = self.load_epochs(self.biodata["stimuli"], start)
        return {"cor": cor,
                "zyg": zyg,
                "eda": eda,
                "ecg": ecg,
                "stimuli": stimuli}

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
            singal_var_second_moment = np.sqrt(singal_variation_second_moment)
            singal_mean_val_first_diff = np.mean(signal[1:]-signal[:-1])
            singal_mean_abs_val_first_diff = np.mean(np.abs(signal[1:]-signal[:-1]))
            singal_mean_abs_val_second_diff = np.mean(np.abs(signal[2:]-signal[:-2]))

            tmp.append([singal_max, singal_min, singal_mean, singal_std, singal_var, singal_rms, singal_power, singal_peak, singal_p2p, singal_skew,
                        singal_kurtosis, singal_crestfactor, singal_formfactor, singal_pulseindicator, singal_var_second_moment, singal_variation_second_moment,
                        singal_var_second_moment, singal_mean_val_first_diff, singal_mean_abs_val_first_diff, singal_mean_abs_val_second_diff])
            col = [name+"max", name+"min", name+"mean", name+"std", name+"var", name+"rms", name+"power",
                                name+"peak", name+"p2p", name+"skew", name+"kurtosis", name+"crestfactor", name+"formfactor",
                                name+"pulseindicator", name+"var_second_moment", name+"variation_second_moment", name+"var_second_moment",
                                name+"mean_val_first_diff", name+"mean_abs_val_first_diff", name+"mean_abs_val_second_diff"]

        return pd.DataFrame(data = tmp, columns=col)

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

    def generate_eda_features(self, epochs):
        eda_features_tonic = self.calculate_statistical_features(epochs, name="eda_tonic_", key="EDA_Tonic")
        eda_features_phasic = self.calculate_statistical_features(epochs, name="eda_phasic_", key="EDA_Phasic")
        eda_features_clean = self.calculate_statistical_features(epochs, name="eda_clean_", key="EDA_Clean")
        generated_feature = pd.concat([eda_features_tonic.reset_index(drop=True), eda_features_phasic.reset_index(drop=True), eda_features_clean.reset_index(drop=True)], axis=1)
        analyzed_eda = nk.eda_analyze(epochs)
        analyzed_eda = analyzed_eda[["Condition", "EDA_Peak_Amplitude"]]
        return pd.concat([analyzed_eda.reset_index(drop=True), generated_feature.reset_index(drop=True)], axis=1)

    def generate_emg_features(self, epochs, name="emg_"):
        emg_features_clean = self.calculate_statistical_features(epochs, name=name+"_clean_", key="EMG_Clean")
        emg_features_amplitude = self.calculate_statistical_features(epochs, name=name+"_amplitude_", key="EMG_Amplitude")
        emg_frequ_features = self.generate_frequence_features(epochs, name=name+"_", key="EMG_Clean")
        return pd.concat([emg_features_clean.reset_index(drop=True), emg_features_amplitude.reset_index(drop=True), emg_frequ_features.reset_index(drop=True)], axis=1)

    def generate_ecg_features(self, epochs):
        ecg_features_hr = self.calculate_statistical_features(epochs, name="ecg_hr_", key="ECG_Rate")
        ecg_features_clean = self.calculate_statistical_features(epochs, name="ecg_clean_", key="ECG_Clean")
        analyzed_ecg = nk.ecg_analyze(epochs)
        analyzed_ecg = analyzed_ecg.drop(["Condition", "Label", "ECG_Phase_Atrial", "ECG_Phase_Ventricular", "ECG_Quality_Mean", "Event_Onset"], axis=1)
        return pd.concat([analyzed_ecg.reset_index(drop=True), ecg_features_hr.reset_index(drop=True), ecg_features_clean.reset_index(drop=True)], axis=1)

    def generate_all_features(self, epochs, pain=1, subject="S001"):
        ecg = self.generate_ecg_features(epochs["ecg"])
        zyg = self.generate_emg_features(epochs["zyg"], "zyg")
        cor = self.generate_emg_features(epochs["cor"], "cor")
        eda = self.generate_eda_features(epochs["eda"])
        #add meta information
        ecg["subject"] = subject
        ecg["pain"] = pain
        return pd.concat([ecg.reset_index(drop=True), eda.reset_index(drop=True), zyg.reset_index(drop=True), cor.reset_index(drop=True)], axis=1)

    def generate_pain_no_pain_features(self, pain=2, nopain=-4):
        epochs_pain = self.generate_epochs(pain)
        epochs_nopain = self.generate_epochs(nopain)

        all_features_pain = self.generate_all_features(epochs_pain, pain=1, subject=self.subject)
        all_features_nopain = self.generate_all_features(epochs_nopain, pain=0, subject=self.subject)
        all_features = all_features_pain.append(all_features_nopain, ignore_index=True)
        return all_features

    def save_features(self, features):
        features.to_pickle(self.saving_folder+self.subject+".pkl")