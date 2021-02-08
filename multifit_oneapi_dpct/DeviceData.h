#ifndef DeviceData_h
#define DeviceData_h

#include "EigenMatrixTypes.h"

struct DoFitArgs {
  SampleVector samples;
  SampleMatrix samplecor;
  double pederr;
  BXVector bxs;
  FullSampleVector fullpulse;
  FullSampleMatrix fullpulsecov;

  DoFitArgs(const SampleVector& samples,
            const SampleMatrix& samplecor,
            double pederr,
            const BXVector& bxs,
            const FullSampleVector& fullpulse,
            const FullSampleMatrix& fullpulsecov)
      : samples(samples),
        samplecor(samplecor),
        pederr(pederr),
        bxs(bxs),
        fullpulse(fullpulse),
        fullpulsecov(fullpulsecov) {}
};

struct Output {
  double chi2;
  double ampl;
  int status;
//   std::vector<double> v_amplitudes;
  BXVector BXs;
  PulseVector X;
};

struct DoFitResults {
  double chisq;
  BXVector BXs;
  PulseVector X;
  bool status;
};

#endif
