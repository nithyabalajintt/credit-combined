import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { ThumbsUp, ThumbsDown, CheckCircle, XCircle, AlertCircle, ClipboardEdit, Loader2 } from 'lucide-react';
 
// ... keep all existing interfaces and utility functions the same ...
interface AssessmentData {
  assessmentId: number;
  score: number;
  riskNarrative: string;
  companyName: string;
  loanDetails: {
    amount: string;
    term: string;
    purpose: string;
  };
  feedbackCount: number;
}
 
const formatAnalysisText = (text: string) => {
  // Clean the text first
  const cleanedText = text
    .replace(/^#+\s*/gm, '')    // Remove markdown headers
    .replace(/\*\*/g, '')       // Remove bold markers
    .replace(/`{3}/g, '')       // Remove code blocks
    .replace(/\n{3,}/g, '\n\n'); // Reduce excessive newlines
 
  // Split into paragraphs and create JSX elements
  return cleanedText.split('\n\n').map((paragraph, index) => (
    <p key={index} className="text-slate-300 mb-4 leading-relaxed">
      {paragraph}
    </p>
  ));
};
 
function ResultPage() {
  // ... keep all existing state and logic the same ...
  const navigate = useNavigate();
  const { state } = useLocation();
  const [feedback, setFeedback] = useState<'up' | 'down' | null>(null);
  const [feedbackText, setFeedbackText] = useState('');
  const [assessmentData, setAssessmentData] = useState<AssessmentData | null>(null);
  const [isRegenerating, setIsRegenerating] = useState(false);
  const [apiError, setApiError] = useState('');
 
  useEffect(() => {
    if (!state) {
      navigate('/');
    } else {
      setAssessmentData({
        assessmentId: state.assessmentId,
        score: state.score,
        riskNarrative: state.riskNarrative,
        companyName: state.companyName,
        loanDetails: state.loanDetails,
        feedbackCount: state.feedbackCount || 0
      });
    }
  }, [state, navigate]);
 
  const getScoreColor = (score: number) => {
    if (score <= 30) return 'text-green-600';
    if (score <= 70) return 'text-yellow-600';
    return 'text-red-600';
  };
  const getRiskCategory = (score: number) => {
    if (score <= 30) return 'Low Risk';
    if (score <= 70) return 'Medium Risk';
    return 'High Risk';
  };
 
  const handleFeedbackSubmit = async () => {
    if (!feedbackText.trim() || !assessmentData) return;
 
    try {
      setIsRegenerating(true);
      setApiError('');
 
      // Submit feedback
      const feedbackResponse = await fetch(
        `http://localhost:5000/api/assessments/${assessmentData.assessmentId}/feedback`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: feedbackText })
        }
      );
 
      if (!feedbackResponse.ok) {
        const errorData = await feedbackResponse.json();
        throw new Error(errorData.error || 'Feedback submission failed');
      }
 
      // Regenerate narrative
      const regenerateResponse = await fetch(
        `http://localhost:5000/api/assessments/${assessmentData.assessmentId}/narrative`,
        { method: 'POST' }
      );
 
      if (!regenerateResponse.ok) {
        const errorData = await regenerateResponse.json();
        throw new Error(errorData.error || 'Regeneration failed');
      }
 
      const newData = await regenerateResponse.json();
      setAssessmentData(prev => ({
        ...prev!,
        score: newData.score,
        riskNarrative: newData.narrative,
        feedbackCount: newData.feedbackCount
      }));
      setFeedback(null);
      setFeedbackText('');
 
    } catch (err) {
      setApiError(err instanceof Error ? err.message : 'Request failed');
    } finally {
      setIsRegenerating(false);
    }
  };
 
  const handleApproveLoan = async () => {
    if (!assessmentData) return;
 
    try {
      const response = await fetch(
        `http://localhost:5000/api/assessments/${assessmentData.assessmentId}/credit-note`,
        { method: 'POST' }
      );
 
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Credit note generation failed');
      }
 
      const creditNoteData = await response.json();
      navigate('/credit-note', {
        state: {
          creditNote: creditNoteData.credit_note,
          company: creditNoteData.company,
          loanDetails: creditNoteData.loan_details,
          assessmentId: assessmentData.assessmentId
        }
      });
 
    } catch (err) {
      setApiError(err instanceof Error ? err.message : 'Approval failed');
    }
  };  
  if (!assessmentData) return (
    <div className="min-h-screen bg-white flex items-center justify-center">
      <Loader2 className="w-12 h-12 text-blue-600 animate-spin" />
    </div>
  );
 
  return (
    <div className="min-h-screen bg-white">
      {/* Navigation Header */}
      <div className="border-b border-gray-200">
        <div className="container mx-auto px-4 py-4">
          <div className="flex justify-between items-center">
            <h1 className="text-2xl font-bold text-blue-900">CREDIT</h1>
            <nav className="flex gap-8 items-center">
              <a href="#" className="text-gray-600 hover:text-blue-900">Investment Bank</a>
              <a href="#" className="text-gray-600 hover:text-blue-900">About us</a>
              <a href="#" className="text-gray-600 hover:text-blue-900">Careers</a>
              <button className="bg-blue-900 text-white px-6 py-2 rounded-full hover:bg-blue-800">
                Approve Loan
              </button>
            </nav>
          </div>
        </div>
      </div>
 
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {apiError && (
            <div className="mb-6 p-4 bg-red-100 text-red-700 rounded-lg border border-red-200">
              <AlertCircle className="inline mr-2" />
              {apiError}
            </div>
          )}
 
          {/* Main Content Card */}
          <div className="bg-white rounded-xl p-8 shadow-lg border border-gray-200">
            {/* Score Header */}
            <div className="text-center mb-8">
              <h1 className="text-3xl font-bold text-gray-900 mb-4">
                {assessmentData.companyName} Credit Assessment
              </h1>
              <div className="flex flex-col items-center">
                <div className={`text-6xl font-bold ${getScoreColor(assessmentData.score)} mb-2`}>
                  {assessmentData.score}
                  <span className="text-2xl ml-2 text-gray-500">/100</span>
                </div>
                <div className="text-lg font-semibold text-gray-700">
                  {getRiskCategory(assessmentData.score)} Category
                </div>
              </div>
            </div>
 
            {/* Loan Details */}
            <div className="grid grid-cols-3 gap-4 mb-8 bg-gray-50 p-4 rounded-lg">
              <div className="text-center">
                <div className="text-sm text-gray-500">Loan Amount</div>
                <div className="text-lg text-gray-900 font-medium">
                  ${Number(assessmentData.loanDetails.amount).toLocaleString()}
                </div>
              </div>
              <div className="text-center">
                <div className="text-sm text-gray-500">Term</div>
                <div className="text-lg text-gray-900 font-medium">
                  {assessmentData.loanDetails.term} months
                </div>
              </div>
              <div className="text-center">
                <div className="text-sm text-gray-500">Purpose</div>
                <div className="text-lg text-gray-900 font-medium capitalize">
                  {assessmentData.loanDetails.purpose.toLowerCase()}
                </div>
              </div>
            </div>
 
            {/* Risk Analysis Section */}
            <div className="mb-8 bg-gray-50 p-6 rounded-lg">
              <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                <ClipboardEdit className="w-5 h-5 mr-2 text-blue-600" />
                AI Risk Analysis
              </h3>
              <div className="text-gray-700 leading-relaxed space-y-4">
                {formatAnalysisText(assessmentData.riskNarrative)}
              </div>
            </div>
 
            {/* Feedback Section */}
            <div className="border-t border-gray-200 pt-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                {feedback === null && 'How would you rate this analysis?'}
                {feedback === 'down' && 'What needs improvement?'}
                {feedback === 'up' && 'Final Loan Decision'}
              </h3>
 
              {feedback === null && (
                <div className="flex justify-center space-x-4">
                  <button
                    onClick={() => setFeedback('up')}
                    className="p-3 rounded-full bg-gray-100 hover:bg-gray-200 transition transform hover:scale-110"
                  >
                    <ThumbsUp className="w-6 h-6 text-green-600" />
                  </button>
                  <button
                    onClick={() => setFeedback('down')}
                    className="p-3 rounded-full bg-gray-100 hover:bg-gray-200 transition transform hover:scale-110"
                  >
                    <ThumbsDown className="w-6 h-6 text-red-600" />
                  </button>
                </div>
              )}
 
              {feedback === 'down' && (
                <div className="space-y-4">
                  <textarea
                    value={feedbackText}
                    onChange={(e) => setFeedbackText(e.target.value)}
                    placeholder="Specific feedback helps us improve..."
                    className="w-full px-4 py-3 rounded-lg border border-gray-300 text-gray-900 placeholder-gray-400 focus:ring-2 focus:ring-blue-400 focus:border-transparent transition-all"
                    rows={3}
                    disabled={isRegenerating}
                  />
                  <div className="flex justify-end space-x-3">
                    <button
                      onClick={() => setFeedback(null)}
                      className="px-4 py-2 text-gray-600 hover:text-gray-800 disabled:opacity-50"
                      disabled={isRegenerating}
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleFeedbackSubmit}
                      disabled={isRegenerating || !feedbackText.trim()}
                      className="px-6 py-2 bg-blue-900 text-white rounded-lg hover:bg-blue-800 disabled:bg-blue-400 flex items-center"
                    >
                      {isRegenerating && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                      {isRegenerating ? 'Processing...' : 'Submit & Improve'}
                    </button>
                  </div>
                </div>
              )}
 
              {feedback === 'up' && (
                <div className="grid grid-cols-2 gap-4">
                  <button
                    onClick={handleApproveLoan}
                    className="p-4 bg-blue-900 text-white rounded-lg flex flex-col items-center justify-center space-y-2 transition hover:bg-blue-800"
                  >
                    <CheckCircle className="w-8 h-8 text-white" />
                    <span className="font-semibold">Approve Loan</span>
                    <span className="text-sm text-blue-100">Generate Credit Note</span>
                  </button>
                  <button
                    onClick={() => navigate('/')}
                    className="p-4 bg-red-900 text-white rounded-lg flex flex-col items-center justify-center space-y-2 transition hover:bg-red-800"
                  >
                    <XCircle className="w-8 h-8 text-white" />
                    <span className="font-semibold">Reject Application</span>
                    <span className="text-sm text-red-100">Return to Dashboard</span>
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
 
export default ResultPage;
 