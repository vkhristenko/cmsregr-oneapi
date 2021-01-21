//
// Takes waveforms from the "file".
// Creates NSAMPLES starting with IDSTART time with the step of NFREQ ns
// Applies noise (correlated) for each sample
// Stores samples and true in-time amplitude
// To run:
// > root -l -q CreateData.C+
//

// repo includes
#include "Pulse.h"

// root includes
#include <TF1.h>
#include <TFile.h>
#include <TGraph.h>
#include <TH1.h>
#include <TMath.h>
#include <TProfile.h>
#include <TRandom.h>
#include <TTree.h>
#include <iostream>

int main(int argc, char** argv) {
  TRandom rnd;

  // Default variables
  // time shift in ns of pulse
  float pulse_shift = 0;

  // time shift in ns of pileup
  float pileup_shift = 0;

  // number of events to generate
  int nEventsTotal = 1000;

  // number of samples per impulse
  int NSAMPLES = 10;

  // number of samples per impulse
  float NFREQ = 25;

  // number of pile up
  float nPU = 0;

  // signal amplitude in GeV
  float signalAmplitude = 10.0;

  // Noise level (GeV)
  float sigmaNoise = 0.044;
  float sigmaNoiseScale = 1;

  // PU Scale factor
  float puFactor = 1;

  // total number of bunches in "LHC" bunch train
  int NBXTOTAL = 2800;

  // CRRC shaping time in ns. For QIE, set it to 1e-1
  float pulse_tau = 43;

  // Different eta regions have different pileup energy distributions (saved in the root file)
  const float eta = 0.0;

  // pedestal shift in GeV
  float pedestal = 0.0;

  char* wf_name;

  // Changing variables if passed in on the command line
  if (argc >= 2) // temporal_shift
    pulse_shift = atof(argv[1]);
  if (argc >= 3) // number_of_events
    nEventsTotal = atoi(argv[2]);
  if (argc >= 4) // NSAMPLES
    NSAMPLES = atoi(argv[3]);
  if (argc >= 5) // NFREQ
    NFREQ = atof(argv[4]);
  if (argc >= 6) // nPU
    nPU = atof(argv[5]);
  if (argc >= 7) // signalAmplitude
    signalAmplitude = atof(argv[6]);
  if (argc >= 8) // sigmaNoise
    sigmaNoiseScale = atof(argv[7]);
  if (argc >= 9) // puFactor
    puFactor = atof(argv[8]);
  if (argc >= 10) // wf_name_string
  {
    wf_name = argv[9];
  } else {
    std::string wf_name_string = "CRRC43";
    wf_name = (char*)wf_name_string.c_str();
  }
  if (argc >= 11) // pu_shift
    pileup_shift = atof(argv[10]);

  sigmaNoise = sigmaNoise * sigmaNoiseScale;

  //---- fix the correct BX
  float real_pulse_shift = pulse_shift;
  //   pulse_shift += 25.;

  // Makeshift way of passing in an option to manually set noise correlations.
  // Currently, this option will be ignored unless a 0 or a 1 is passed to it.
  // May add more flexibility in the future.
  float correlation_flag = 0.5;
  if (argc >= 12) //  noise-correlation
    correlation_flag = atof(argv[11]);

  if (argc >= 13) // pedestal
    pedestal = atof(argv[12]);

  //---- fix the correct BX
  //   int IDSTART = 7*25;
  int IDSTART = 6 * 25;
  int WFLENGTH = 500 * 4;  // step 1/4 ns in waveform
  if ((IDSTART + NSAMPLES * NFREQ) > 500) {
    WFLENGTH = (IDSTART + NSAMPLES * NFREQ) * 4 + 100;
  }

  //---- distortion of the 4th sample to simulate slew rate effect in pre-amp
  //----    1. --> 100%, meaning no distortion
  //----    0.90 --> 90%, meaning the point is multiplied by 0.90
  float distortion_sample_4 = 1.;
  if (argc >= 14) // distortion_sample_4
    distortion_sample_4 = atof(argv[13]);

  std::cout << " NSAMPLES = " << NSAMPLES << std::endl;
  std::cout << " NFREQ = " << NFREQ << std::endl;
  std::cout << " nPU = " << nPU << std::endl;
  std::cout << " signalAmplitude = " << signalAmplitude << std::endl;
  std::cout << " Generation of digitized samples " << std::endl;
  std::cout << " WFLENGTH = " << WFLENGTH << std::endl;
  std::cout << " nEventsTotal = " << nEventsTotal << std::endl;
  std::cout << " pedestal = " << pedestal << std::endl;
  std::cout << " wf_name = " << wf_name << std::endl;
  std::cout << " puFactor = " << puFactor << std::endl;
  std::cout << " sigmaNoise = " << sigmaNoise << std::endl;
  std::cout << " sigmaNoiseScale = " << sigmaNoiseScale << std::endl;
  std::cout << " distortion_sample_4 = " << distortion_sample_4 << std::endl;

  Pulse pSh;
  pSh.SetNSAMPLES(NSAMPLES);
  pSh.SetNFREQ(NFREQ);
  pSh.SetIDSTART(IDSTART);
  pSh.SetWFLENGTH(WFLENGTH);

  // make sure these inputs are what you really want
  // TFile *file = new TFile("data/EmptyFileCRRC43.root");
  //   std::string wf_file_name = ((std::string) "../data/EmptyFile") +
  //   ((std::string) wf_name) + ((std::string) ".root");
  std::string wf_file_name =
      ((std::string) "../data/EmptyFile") + ((std::string) ".root");

  std::cout << " wf_file_name = " << wf_file_name << std::endl;

  pSh.SetFNAMESHAPE(wf_file_name);
  pSh.Init();
  // Get the value of tau form the (initialized!) pulse
  pulse_tau = pSh.tau();

  // Change noise correlations to max/zero if one of the special flags was set
  if (correlation_flag == 0.0) {
    pSh.SetNoiseCorrelationZero();
  } else if (correlation_flag == 1.0) {
    pSh.SetNoiseCorrelationMax();
  }

  TFile* file = new TFile(wf_file_name.c_str());
  TString filenameOutput;
  if (correlation_flag == 0.0) {
    filenameOutput = Form(
        "../data/"
        "mysample_%d_%.3f_%.3f_%d_%.2f_%.2f_%.2f_%.3f_%.2f_NoiseUncorrelated."
        "root",
        nEventsTotal, real_pulse_shift, pileup_shift, NSAMPLES, NFREQ,
        signalAmplitude, nPU, sigmaNoiseScale, puFactor);
  } else if (correlation_flag == 1.0) {
    filenameOutput = Form(
        "../data/"
        "mysample_%d_%.3f_%.3f_%d_%.2f_%.2f_%.2f_%.3f_%.2f_"
        "NoiseFullyCorrelated.root",
        nEventsTotal, real_pulse_shift, pileup_shift, NSAMPLES, NFREQ,
        signalAmplitude, nPU, sigmaNoiseScale, puFactor);
  } else {
    filenameOutput = Form(
        "../data/"
        "mysample_%d_%.3f_%.3f_%d_%.2f_%.2f_%.2f_%.3f_%.2f_%.2f_slew_%.2f.root",
        nEventsTotal, real_pulse_shift, pileup_shift, NSAMPLES, NFREQ,
        signalAmplitude, nPU, sigmaNoiseScale, puFactor, pedestal,
        distortion_sample_4);
  }
  TFile* fileOut = new TFile(filenameOutput.Data(), "recreate");

  std::cout << " filenameOutput = " << filenameOutput.Data() << std::endl;

  // Get PDF for pileup
  int indx = 10 * fabs(eta) / 0.1;
  if (indx < 0)
    indx = 0;
  if (indx > 13)
    indx = 13;
  char hname[120];
  sprintf(hname, "PileupPDFs/pupdf_%d", indx);
  TH1D* pupdf = (TH1D*)file->Get(hname);
  pupdf->SetDirectory(0);

  // Variables for filling the tree
  int nSmpl = NSAMPLES;
  float nFreq = NFREQ;
  double amplitudeTruth;
  std::vector<double> samples;
  std::vector<double> samples_noise;
  int BX0;
  int nBX = NBXTOTAL;
  int nWF = WFLENGTH;
  std::vector<int> nMinBias;
  std::vector<double> energyPU;  // along a complete LHC circle
  std::vector<double> pulse_signal;
  std::vector<double> pileup_signal;
  double signalTruth = signalAmplitude;

  // Making the tree
  TTree* treeOut = new TTree("Samples", "");
  treeOut->Branch("pulse_shift", &real_pulse_shift, "pulse_shift/F");
  treeOut->Branch("pileup_shift", &pileup_shift, "pileup_shift/F");
  treeOut->Branch("nSmpl", &nSmpl, "nSmpl/I");
  treeOut->Branch("nFreq", &nFreq, "nFreq/F");
  treeOut->Branch("amplitudeTruth", &amplitudeTruth, "amplitudeTruth/D");
  treeOut->Branch("samples", &samples);
  treeOut->Branch("samples_noise", &samples_noise);
  treeOut->Branch("nPU", &nPU, "nPU/F");
  treeOut->Branch("BX0", &BX0, "BX0/I");
  treeOut->Branch("nBX", &nBX, "nBX/I");
  treeOut->Branch("nWF", &nWF, "nWF/I");
  treeOut->Branch("nMinBias", &nMinBias);
  treeOut->Branch("energyPU", &energyPU);
  treeOut->Branch("pulse_signal", &pulse_signal);
  treeOut->Branch("pileup_signal", &pileup_signal);
  treeOut->Branch("signalTruth", &signalTruth, "signalTruth/D");
  treeOut->Branch("sigmaNoise", &sigmaNoise, "sigmaNoise/F");
  treeOut->Branch("puFactor", &puFactor, "puFactor/F");
  treeOut->Branch("pulse_tau", &pulse_tau, "pulse_tau/F");
  // wf_name is already a pointer (char *) so we don't need to use &
  treeOut->Branch("wf_name", wf_name, "wf_name/C");
  treeOut->Branch("input_pedestal", &pedestal, "input_pedestal/F");
  treeOut->Branch("distortion_sample_4", &distortion_sample_4,
                  "distortion_sample_4/F");

  for (int ievt = 0; ievt < nEventsTotal; ievt++) {
    if (!(ievt % 100)) {
      std::cout << " ievt = " << ievt << " :: " << nEventsTotal << std::endl;
    }

    nMinBias.clear();
    energyPU.clear();
    for (int ibx = 0; ibx < nBX; ibx++) {
      // number of min-bias interactions in each bunch crossing
      nMinBias.push_back(rnd.Poisson(nPU));

      // total energy per BX
      energyPU.push_back(0.);
      for (int imb = 0; imb < nMinBias.at(ibx); imb++) {
        energyPU.at(ibx) += pow(10., pupdf->GetRandom());
      }

      // pick in-time BX
      BX0 = int(nBX * rnd.Rndm());
      while (BX0 > (nBX - 3 * NSAMPLES) ||
             BX0 < (3 * NSAMPLES)) {  // ---- 15 or 11 ?
        BX0 = int(nBX * rnd.Rndm());
      }
    }

    // Initialize the pulse and pileup signals to be zero everwhere
    pulse_signal.clear();
    for (int iwf = 0; iwf < nWF; iwf++) {
      pulse_signal.push_back(0.);
    }
    pileup_signal.clear();
    for (int iwf = 0; iwf < nWF; iwf++) {
      pileup_signal.push_back(0.);
    }

    // Add pileup to the waveform
    // time window is nWF ns wide and is centered at BX0
    for (int ibx = 0; ibx < nBX; ibx++) {
      for (int iwf = 0; iwf < nWF; iwf++) {
        double t = (BX0 - ibx) * 25. + iwf / 4. - (500 / 2) + 25.;
        double temp = pileup_signal.at(iwf);
        // adding the pu times the scale factor to the waveform
        pileup_signal.at(iwf) = temp + energyPU.at(ibx) * pSh.fShape(t) * puFactor;            
//         std::cout << " pileup_signal [" << iwf << "] = " << pileup_signal.at(iwf) << std::endl;    
      }
    }

    // Add signal to the waveform
    for (int iwf = 0; iwf < nWF; iwf++) {
      pulse_signal.at(iwf) += signalTruth * pSh.fShape(iwf / 4. - (500 / 2) + 25.);
    }

    // Construct the digitized points
    std::vector<double> samplesUncorrelated;
    for (int i = 0; i < NSAMPLES; ++i) {
      samplesUncorrelated.push_back(rnd.Gaus(0, 1));
    }

    // Noise correlations
    samples_noise.clear();
    samples.clear();
    for (int i = 0; i < NSAMPLES; ++i) {
      samples.push_back(0);
      for (int j = 0; j < NSAMPLES; ++j) {
        samples.at(i) += pSh.cholesky(i, j) * samplesUncorrelated.at(j);
      }
    }

    for (int i = 0; i < NSAMPLES; ++i) {
      samples.at(i) *= sigmaNoise;
    }

    for (int i = 0; i < NSAMPLES; ++i) {
      samples_noise.push_back(samples.at(i));
    }

    // Add signal and pileup
    for (int i = 0; i < NSAMPLES; ++i) {
      int pulse_index = TMath::Nint(4 * (IDSTART + i * NFREQ - pulse_shift));

      //---- slew rate
      if (distortion_sample_4 != 1) {
        if (i==4) samples.at(i) += distortion_sample_4 * pulse_signal.at(pulse_index);
        else      samples.at(i) +=                       pulse_signal.at(pulse_index);        
      }
      else {
        samples.at(i) += pulse_signal.at(pulse_index);
      }
      
      int pileup_index = TMath::Nint(4*(IDSTART + i * NFREQ - pileup_shift));
      samples.at(i) += pileup_signal.at(pileup_index);
      
    }

    // Add pedestal
    for (int i = 0; i < NSAMPLES; ++i) {
      samples.at(i) += pedestal;
    }

    // Adding energyPU to the true amplitude
    amplitudeTruth = signalTruth + energyPU.at(BX0);

    treeOut->Fill();
  }

  treeOut->Write();
  fileOut->Close();
  file->Close();

  std::cout << " output file = " << filenameOutput.Data() << std::endl;
}
