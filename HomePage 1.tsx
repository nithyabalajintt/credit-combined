import { useState, ChangeEvent, FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, FileText, AlertCircle, Gauge, Loader2 } from 'lucide-react';

interface FormData {
  companyName: string;
  loanAmount: string;
  loanTerm: string;
  loanPurpose: string;
  collateralAmount: string;
  creditScore: string;
  file: File | null;
}

interface AssessmentResponse {
  id: number;
  company_name: string;
  loan_amount: number;
  loan_term: number;
  loan_purpose: string;
  collateral_amount: number;
  credit_score: number;
}

interface NarrativeResponse {
  narrative: string;
  score: number;
  feedbackCount: number;
}

function HomePage() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState<FormData>({
    companyName: '',
    loanAmount: '100000000',
    loanTerm: '12',
    loanPurpose: 'Working capital',
    collateralAmount: '120000000',
    creditScore: '400',
    file: null,
  });
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [apiError, setApiError] = useState('');
  const [uploadMessage, setUploadMessage] = useState('');
  const [isUploadError, setIsUploadError] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  const validateForm = () => {
    const errors: Record<string, string> = {};
    if (!formData.companyName.trim()) {
      errors.companyName = 'Company name is required';
    }
    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const formData = new FormData();
      formData.append('file', file);

      setUploadMessage('');
      setIsUploadError(false);
      setIsUploading(true);

      try {
        const response = await fetch('http://localhost:5000/upload-file', {
          method: 'POST',
          body: formData,
        });

        const data = await response.json();
        if (response.ok) {
          setUploadMessage(data.message);
          setIsUploadError(false);
        } else {
          setUploadMessage(data.error || 'Upload failed');
          setIsUploadError(true);
        }
      } catch (error) {
        setUploadMessage(`Network Error: ${(error as Error).message}`);
        setIsUploadError(true);
      } finally {
        setIsUploading(false);
      }
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!validateForm()) return;

    setIsSubmitting(true);
    setApiError('');

    try {
      const assessmentResponse = await fetch('http://localhost:5000/api/assessments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          companyName: formData.companyName,
          loanAmount: parseFloat(formData.loanAmount),
          loanTerm: parseInt(formData.loanTerm),
          loanPurpose: formData.loanPurpose,
          collateralAmount: parseFloat(formData.collateralAmount),
          creditScore: parseInt(formData.creditScore)
        })
      });

      if (!assessmentResponse.ok) {
        const errorData = await assessmentResponse.json();
        throw new Error(errorData.error || 'Failed to create assessment');
      }

      const assessmentData: AssessmentResponse = await assessmentResponse.json();

      const narrativeResponse = await fetch(
        `http://localhost:5000/api/assessments/${assessmentData.id}/narrative`,
        { method: 'POST' }
      );

      if (!narrativeResponse.ok) {
        const errorData = await narrativeResponse.json();
        throw new Error(errorData.error || 'Failed to generate narrative');
      }

      const narrativeData: NarrativeResponse = await narrativeResponse.json();

      navigate('/result', {
        state: {
          assessmentId: assessmentData.id,
          score: narrativeData.score,
          riskNarrative: narrativeData.narrative,
          companyName: formData.companyName,
          loanDetails: {
            amount: formData.loanAmount,
            term: formData.loanTerm,
            purpose: formData.loanPurpose,
            collateral: formData.collateralAmount,
            creditScore: formData.creditScore
          }
        }
      });

    } catch (err) {
      setApiError(err instanceof Error ? err.message : 'Failed to start assessment');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-cover bg-center" style={{ 
      backgroundImage: "url('https://images.unsplash.com/photo-1601597111158-2fceff292cdc?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80')",
      backgroundAttachment: 'fixed'
    }}>
      <div className="container mx-auto px-4 py-12">
        <div className="max-w-2xl mx-auto">
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold text-white mb-4">
              Loan Application Interface
            </h1>
            <p className="text-white/80 text-lg">
              Complete your financial profile to apply for a loan
            </p>
          </div>

          {apiError && (
            <div className="mb-6 p-4 bg-red-100 text-red-700 rounded-lg border border-red-200">
              {apiError}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="bg-white rounded-xl p-8 shadow-2xl border border-gray-200">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                {/* Company Name */}
                <div>
                  <label className="block text-gray-800 mb-2 font-medium">
                    Company Name *
                  </label>
                  <input
                    type="text"
                    className={`w-full px-4 py-3 rounded-lg border ${
                      validationErrors.companyName ? 'border-red-400' : 'border-gray-300'
                    } text-gray-900 placeholder-gray-400 focus:ring-2 focus:ring-red-500 focus:border-transparent transition-all`}
                    placeholder="Enter company name"
                    value={formData.companyName}
                    onChange={(e) => setFormData(prev => ({ ...prev, companyName: e.target.value }))}
                  />
                  {validationErrors.companyName && (
                    <span className="text-red-500 text-sm mt-1">
                      {validationErrors.companyName}
                    </span>
                  )}
                </div>

                {/* Collateral Amount */}
                <div>
                  <label className="block text-gray-800 mb-2 font-medium">
                    Collateral Amount (Cr)
                  </label>
                  <div className="relative">
                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400">₹</span>
                    <input
                      type="number"
                      className="w-full pl-8 pr-4 py-3 rounded-lg border border-gray-300 text-gray-900 placeholder-gray-400 focus:ring-2 focus:ring-red-500 focus:border-transparent transition-all"
                      value={formData.collateralAmount}
                      onChange={(e) => setFormData(prev => ({ ...prev, collateralAmount: e.target.value }))}
                    />
                  </div>
                </div>
              </div>

              {/* File Upload */}
              <div className="mb-6">
                <label className="block text-gray-800 mb-2 font-medium">
                  Upload Financial Documents
                </label>
                <div className="relative">
                  <input
                    type="file"
                    accept=".pdf"
                    onChange={handleFileChange}
                    className="hidden"
                    id="file-upload"
                  />
                  <label
                    htmlFor="file-upload"
                    className={`flex items-center justify-center w-full px-4 py-6 rounded-lg border-2 border-dashed ${
                      isUploadError ? 'border-red-400' : 'border-gray-300'
                    } cursor-pointer transition-colors bg-gray-50`}
                  >
                    <div className="text-center">
                      {isUploading ? (
                        <Loader2 className="w-8 h-8 text-gray-400 mx-auto mb-2 animate-spin" />
                      ) : (
                        <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                      )}
                      <p className="text-sm text-gray-500">
                        {isUploading ? 'Uploading...' : 'Drag & drop or click to upload PDF'}
                      </p>
                    </div>
                  </label>

                  {uploadMessage && (
                    <div className={`mt-2 p-2 rounded-lg ${
                      isUploadError
                        ? 'bg-red-100 text-red-700 border border-red-200'
                        : 'bg-red-100 text-green-700 border border-red-200'
                    }`}
                    >
                      <div className="flex items-center">
                        {isUploadError ? (
                          <AlertCircle className="w-4 h-4 mr-2" />
                        ) : (
                          <FileText className="w-4 h-4 mr-2" />
                        )}
                        {uploadMessage}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Loan Details */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div>
                  <label className="block text-gray-800 mb-2 font-medium">
                    Loan Amount (Cr)
                  </label>
                  <div className="relative">
                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400">₹</span>
                    <input
                      type="number"
                      className="w-full pl-8 pr-4 py-3 rounded-lg border border-gray-300 text-gray-900 placeholder-gray-400 focus:ring-2 focus:ring-red-500 focus:border-transparent transition-all"
                      value={formData.loanAmount}
                      onChange={(e) => setFormData(prev => ({ ...prev, loanAmount: e.target.value }))}
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-gray-800 mb-2 font-medium">
                    Loan Term (months)
                  </label>
                  <input
                    type="number"
                    className="w-full px-4 py-3 rounded-lg border border-gray-300 text-gray-900 placeholder-gray-400 focus:ring-2 focus:ring-red-500 focus:border-transparent transition-all"
                    value={formData.loanTerm}
                    onChange={(e) => setFormData(prev => ({ ...prev, loanTerm: e.target.value }))}
                  />
                </div>

                <div>
                  <label className="block text-gray-800 mb-2 font-medium">
                    Loan Purpose
                  </label>
                  <select
                    className="w-full px-4 py-3 rounded-lg border border-gray-300 text-gray-900 focus:ring-2 focus:ring-red-500 focus:border-transparent transition-all"
                    value={formData.loanPurpose}
                    onChange={(e) => setFormData(prev => ({ ...prev, loanPurpose: e.target.value }))}
                  >
                    <option value="Working capital">Working Capital</option>
                    <option value="Equipment purchase">Equipment Purchase</option>
                    <option value="Expansion">Business Expansion</option>
                    <option value="Debt refinancing">Debt Refinancing</option>
                  </select>
                </div>

                <div>
                  <label className="block text-gray-800 mb-2 font-medium">
                    Credit Score
                  </label>
                  <input
                    type="number"
                    min="300"
                    max="900"
                    className="w-full px-4 py-3 rounded-lg border border-gray-300 text-gray-900 placeholder-gray-400 focus:ring-2 focus:ring-red-500 focus:border-transparent transition-all"
                    value={formData.creditScore}
                    onChange={(e) => setFormData(prev => ({ ...prev, creditScore: e.target.value }))}
                  />
                </div>
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isSubmitting}
                className="w-full py-4 px-6 rounded-lg bg-red-600 text-white font-semibold text-lg hover:bg-red-700 transition-all duration-200 disabled:opacity-70 disabled:cursor-not-allowed"
              >
                <div className="flex items-center justify-center space-x-2">
                  {isSubmitting ? (
                    <Loader2 className="w-6 h-6 animate-spin" />
                  ) : (
                    <Gauge className="w-6 h-6" />
                  )}
                  <span>
                    {isSubmitting ? 'Generating Assessment...' : 'Generate Risk Assessment'}
                  </span>
                </div>
              </button>
            </div>
          </form>

          {/* Features Section */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
            <div className="p-6 rounded-xl bg-white/90 backdrop-blur-sm border border-gray-200">
              <FileText className="w-8 h-8 text-red-600 mb-4" />
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Secure Process</h3>
              <p className="text-gray-600">Bank-level encryption and security measures</p>
            </div>
            <div className="p-6 rounded-xl bg-white/90 backdrop-blur-sm border border-gray-200">
              <AlertCircle className="w-8 h-8 text-red-600 mb-4" />
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Fast Approval</h3>
              <p className="text-gray-600">Get decisions within 24 hours</p>
            </div>
            <div className="p-6 rounded-xl bg-white/90 backdrop-blur-sm border border-gray-200">
              <Gauge className="w-8 h-8 text-red-600 mb-4" />
              <h3 className="text-xl font-semibold text-gray-900 mb-2">AI Analysis</h3>
              <p className="text-gray-600">Advanced credit risk assessment</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default HomePage;
