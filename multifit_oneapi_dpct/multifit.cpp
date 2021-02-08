#include <chrono>
#include <iostream>

#include "Pulse.h"
#include "PulseChiSqSNNLSWrapper.h"

#include "TF1.h"
#include "TFile.h"
#include "TH2.h"
#include "TProfile.h"
#include "TTree.h"

using namespace std;

Pulse pSh;

FullSampleVector fullpulse(FullSampleVector::Zero());
FullSampleMatrix fullpulsecov(FullSampleMatrix::Zero());
SampleMatrix noisecor(SampleMatrix::Zero());
BXVector activeBX;
SampleVector amplitudes(SampleVector::Zero());

TFile* fout;
TH1D* h01;
TH1D *hAmpl;
TH1D* hDuration;

void initHist(std::string const& out_file) {
  fout = new TFile(out_file.c_str(), "recreate");
  h01 = new TH1D("h01", "dA", 1000, -5, 5);
  hAmpl = new TH1D("reco_ampl", "reco ampl", 100, 5, 15);
  hDuration = new TH1D("Duration", "Duration", 100, 0, 5000);
}

void init(std::string const& out_file) {
  initHist(out_file);

  // intime sample is [2]
  double pulseShapeTemplate[NSAMPLES + 2];
  for (int i = 0; i < (NSAMPLES + 2); i++) {
    double x = double(IDSTART + NFREQ * (i + 3) + NFREQ -
                      500 / 2);  //----> 500 ns is fixed!

    pulseShapeTemplate[i] = pSh.fShape(x);
    std::cout << " >>  pulseShapeTemplate[" << i << "] "
              << pulseShapeTemplate[i] << " at x = " << x << std::endl;
  }
  //  for(int i=0; i<(NSAMPLES+2); i++) pulseShapeTemplate[i] /=
  //  pulseShapeTemplate[2];
  for (int i = 0; i < (NSAMPLES + 2); ++i)
    fullpulse(i + 7) = pulseShapeTemplate[i];

  for (int i = 0; i < NSAMPLES; ++i) {
    for (int j = 0; j < NSAMPLES; ++j) {
      int vidx = std::abs(j - i);
      noisecor(i, j) = pSh.corr(vidx);
    }
  }

  int activeBXs[] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
  activeBX.resize(10);
  for (unsigned int ibx = 0; ibx < 10; ++ibx) {
    activeBX.coeffRef(ibx) = activeBXs[ibx];
  }
  //  activeBX.resize(1);
  //  activeBX.coeffRef(0) = 0;
}

void run(std::string inputFile,
         int max_iterations,
         int entries_per_kernel = 100) {
  //
  // initilaization: read the input and arrange root tree/branches
  //
  TFile* file2 = new TFile(inputFile.c_str());

  std::vector<double>* samples = new std::vector<double>;
  double amplitudeTruth;
  TTree* tree = (TTree*)file2->Get("Samples");
  tree->SetBranchAddress("amplitudeTruth", &amplitudeTruth);
  tree->SetBranchAddress("samples", &samples);
  int nentries = tree->GetEntries();

  float time_shift = 13.;  //---- default is 13
  float pedestal_shift = 0.;

  float return_chi2 = -99;
  float best_pedestal = 0;
  float best_chi2 = 0;

  std::vector<TH1F*> v_pulses;
  std::vector<TH1F*> v_amplitudes_reco;

  fout->cd();
  TTree* newtree = (TTree*)tree->CloneTree(0);  //("RecoAndSim");
  newtree->SetName("RecoAndSim");

  std::vector<double> samplesReco;
  std::vector<std::vector<double> > complete_samplesReco;
  std::vector<double> complete_chi2;
  std::vector<double> complete_pedestal;
  std::vector<double> pulseShapeTemplate;  
  std::vector<int> activeBXs;
  
  int ipulseintime = 0;
  newtree->Branch("chi2", &return_chi2, "chi2/F");
  newtree->Branch("samplesReco", &samplesReco);
  newtree->Branch("ipulseintime", ipulseintime, "ipulseintime/I");

  newtree->Branch("complete_samplesReco", &complete_samplesReco);
  newtree->Branch("complete_chi2", &complete_chi2);
  newtree->Branch("complete_pedestal", &complete_pedestal);
  newtree->Branch("best_pedestal", &best_pedestal, "best_pedestal/F");
  newtree->Branch("best_chi2", &best_chi2, "best_chi2/F");
  newtree->Branch("pulseShapeTemplate",   &pulseShapeTemplate);
  newtree->Branch("activeBXs",     &activeBXs);
  
  for(int i=0; i<(NSAMPLES+7*int(25 /NFREQ)); i++) {
    double x;
    x = double( IDSTART + NFREQ * i + 3*25. - 500 / 2. );  //----> 500 ns is fixed!  
    pulseShapeTemplate.push_back( pSh.fShape(x));
  }
  
  int totalNumberOfBxActive = 10;

  for (unsigned int ibx = 0; ibx < totalNumberOfBxActive; ++ibx) {
    samplesReco.push_back(0.);
    activeBXs.push_back( ibx * int(25 /NFREQ) - 5 * int(25 /NFREQ) ); //----> -5 BX are active w.r.t. 0 BX
  }

  v_amplitudes_reco.clear();

  std::cout << "entries: " << nentries << std::endl;
  std::cout << "max_iterations: " << max_iterations << std::endl;
  std::cout << "entries_per_kernel: " << entries_per_kernel << std::endl;

  for (auto it = 0; it < max_iterations; ++it) {
    // vector of input parameters to the kernel
    std::vector<DoFitArgs> vargs;

    for (int ie = 0; ie < entries_per_kernel; ++ie) {
      tree->GetEntry(ie % tree->GetEntries());
      for (int i = 0; i < NSAMPLES; ++i)
        amplitudes[i] = samples->at(i);

      double pedval = 0.;
      double pedrms = 1.0;
      vargs.emplace_back(amplitudes, noisecor, pedrms, activeBX, fullpulse,
                         fullpulsecov);
    }

    std::cout << "iteration: " << it
              << " wrapper start with vargs.size() = " << vargs.size()
              << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto vresults = doFitWrapper(vargs);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "wrapper end with vresults.size() = " << vresults.size()
              << std::endl;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();
    if (it > 0)
      hDuration->Fill(duration);
    else
      std::cout << "skipping first iteration" << std::endl;
    std::cout << "duration = " << duration << std::endl;

  
    if (it == 0){
      int ientry = 0;
      for (auto& results : vresults) {
        tree->GetEntry(ientry);
        
        // std::cout << "status = " << results.status << std::endl;
        // std::cout << "chi2 = " << results.chi2 << std::endl;
        
        // double aMax = status ? pulsefunc.X()[ipulseintime] : 0.;
        double aMax = results.ampl;
        // std::cout << "aMax = " << aMax << std::endl;
        //std::cout << "amplitudeTruth = " << amplitudeTruth << std::endl;
        h01->Fill(aMax - amplitudeTruth);
        hAmpl->Fill(aMax);
        
        //---- all reconstructed pulses
        //       samplesReco = results.v_amplitudes;
        //---- save all reconstructed amplitudes
        samplesReco.clear();
        for (unsigned int ip=0; ip<results.BXs.rows(); ++ip) {
          samplesReco.push_back(0.);
        }
        
        for (unsigned int ip=0; ip<results.BXs.rows(); ++ip) {
          samplesReco[ (int(results.BXs.coeff(ip))) + 5] = (results.X)[ ip ];
        }
        
        newtree-> Fill();
        ientry++;  
        
      }
    }
    
  }

  // print some stats
  fout->cd();
  newtree->Write();
  std::cout << "  Mean of REC-MC = " << h01->GetMean() << " GeV" << std::endl;
  std::cout << "  RMS of REC-MC = " << h01->GetRMS() << " GeV" << std::endl;
  std::cout << "  Entries Total = " << h01->GetEntries() << std::endl;
  std::cout << "  Mean Duration = " << hDuration->GetMean() << std::endl;
}

void saveHist() {
  fout->cd();
  h01->Write();
  hAmpl->Write();
  hDuration->Write();
  fout->Close();
}

int main(int argc, char** argv) {
  // default input file
  std::string inputFile = "data/samples_signal_10GeV_pu_0.root";

  // unwrap the cli args
  auto max_iterations = 10;
  auto entries_per_kernel = 100;
  if (argc >= 2)
    inputFile = argv[1];
  if (argc >= 3)
    max_iterations = atoi(argv[2]);
  if (argc >= 4)
    entries_per_kernel = atoi(argv[3]);

  // output
  std::string out_file = "output_gpu.root";

  std::cout << "max_iterations = " << max_iterations
            << "  entries_per_kernel = " << entries_per_kernel << std::endl;

  // start
  std::cout << "1111" << std::endl;
  init(out_file);
  std::cout << "2222" << std::endl;
  run(inputFile, max_iterations, entries_per_kernel);
  std::cout << "3333" << std::endl;
  saveHist();
  std::cout << "4444" << std::endl;

  //std::vector<DoFitArgs> v;
  //auto tmp = doFitWrapper(v);

  return 0;
}
