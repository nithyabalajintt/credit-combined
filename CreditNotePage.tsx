import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { ThumbsUp, ThumbsDown, Download, CheckCircle, Loader2, AlertCircle } from 'lucide-react';



interface CreditNoteData {
  assessmentId: number;
  creditNote: string;
  company: string;
  loanDetails: {
    amount: number;
    term: number;
    purpose: string;
  };
  feedbackCount: number;
}

const formatCreditNoteText = (text: string) => {
  return text.split('\n').map((line, index) => {
    if (!line.trim()) return null;

    if (line.startsWith('## ')) {
      return (
        <h2 key={`h2-${index}`} className="text-xl font-semibold text-red-600 mt-6 mb-3">
          {line.replace(/^##+\s*/, '')}
        </h2>
      );
    }
    if (line.startsWith('### ')) {
      return (
        <h3 key={`h3-${index}`} className="text-lg font-medium text-gray-800 mt-4 mb-2">
          {line.replace(/^###+\s*/, '')}
        </h3>
      );
    }
    if (line.startsWith('**')) {
      return (
        <p key={`strong-${index}`} className="text-black font-medium mb-3">
          {line.replace(/\*\*/g, '')}
        </p>
      );
    }
    return (
      <p key={`p-${index}`} className="text-black mb-3 leading-relaxed">
        {line}
      </p>
    );
  }).filter(Boolean);
};
function CreditNotePage() {
  const navigate = useNavigate();
  const { state } = useLocation();
  const [feedback, setFeedback] = useState<'up' | 'down' | null>(null);
  const [feedbackText, setFeedbackText] = useState('');
  const [creditNoteData, setCreditNoteData] = useState<CreditNoteData | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [apiError, setApiError] = useState('');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!state) {
      navigate('/');
    } else {
      setCreditNoteData({
        assessmentId: state.assessmentId,
        creditNote: state.creditNote,
        company: state.company,
        loanDetails: state.loanDetails,
        feedbackCount: state.feedbackCount || 0
      });
      setIsLoading(false);
    }
  }, [state, navigate]);

  const handleDownload = () => {
    if (!creditNoteData) return;
    const element = document.createElement('a');
    const file = new Blob([creditNoteData.creditNote], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = `${creditNoteData.company}-credit-note.txt`;
    document.body.appendChild(element);
    element.click();
  };

  const handleFeedbackSubmit = async () => {
    if (!creditNoteData || !feedbackText.trim() || creditNoteData.feedbackCount >= 3) return;

    try {
        setIsSubmitting(true);
        setApiError('');

        const response = await fetch(
            `http://localhost:5000/api/assessments/${creditNoteData.assessmentId}/credit-feedback`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: feedbackText })
            }
        );

        const responseData = await response.json();
        if (!response.ok) throw responseData;

        // Update state with new data
        setCreditNoteData(prev => ({
            ...prev!,
            creditNote: responseData.credit_note,
            feedbackCount: responseData.feedback_count
        }));

        setFeedback(null);
        setFeedbackText('');

    } catch (err: any) {
        if (err.error?.includes('Maximum of 3')) {
            setCreditNoteData(prev => ({
                ...prev!,
                feedbackCount: 3
            }));
            setApiError('Maximum feedback submissions (3) reached');
        } else {
            setApiError(err.error || 'Submission failed');
        }
    } finally {
        setIsSubmitting(false);
    }
  };

  if (isLoading) return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 flex items-center justify-center">
      <Loader2 className="w-12 h-12 text-blue-400 animate-spin" />
    </div>
  );

  if (!creditNoteData) return null;

  return (
    <div 
      className="min-h-screen bg-cover bg-center"
      style={{ 
        backgroundImage: "url('https://images.unsplash.com/photo-1601597111158-2fceff292cdc?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80')",
        backgroundAttachment: 'fixed'
      }}
    >
      <div className="container mx-auto px-4 py-12 relative">
        <div className="max-w-4xl mx-auto">
          {apiError && (
            <div className="mb-6 p-4 bg-red-100 text-red-700 rounded-lg border border-red-200">
              <AlertCircle className="inline mr-2" />
              {apiError}
            </div>
          )}

          <div className="bg-white rounded-xl p-8 shadow-2xl border border-gray-200">
            {/* Header */}
            <div className="mb-8 text-center">
              <h1 className="text-3xl font-bold text-gray-900">
                Credit Note for {creditNoteData.company}
              </h1>
              <p className="text-gray-600 mt-2">
                Generated on {new Date().toLocaleDateString()}
                <span className="block text-sm">Feedback iterations: {creditNoteData.feedbackCount} / 3</span>
              </p>
            </div>

            {/* Loan Details */}
            <div className="grid grid-cols-2 gap-4 mb-8">
              <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                <p className="text-gray-600">Loan Amount</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {creditNoteData.loanDetails.amount.toLocaleString()}
                </p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                <p className="text-gray-600">Loan Term</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {creditNoteData.loanDetails.term} months
                </p>
              </div>
            </div>

            {/* Credit Note Content */}
            <div className="mb-8 bg-gray-50 p-6 rounded-lg border border-gray-200 overflow-y-auto max-h-96">
              <div className="whitespace-pre-wrap leading-relaxed">
                {formatCreditNoteText(creditNoteData.creditNote)}
              </div>
            </div>

            {/* Download Section */}
            <div className="mb-8">
              <button
                onClick={handleDownload}
                className="w-full py-3 px-6 bg-red-600 text-white rounded-lg hover:bg-red-700 transition flex items-center justify-center space-x-2"
              >
                <Download className="w-5 h-5" />
                <span>Download Credit Note</span>
              </button>
            </div>

            {/* Feedback Section */}
            <div className="border-t border-gray-200 pt-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Was this credit note helpful?
              </h3>

              {apiError && (
                <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-lg flex items-center">
                  <AlertCircle className="w-5 h-5 mr-2" />
                  {apiError}
                </div>
              )}

              {creditNoteData.feedbackCount >= 3 && (
                <div className="mb-4 p-3 bg-blue-100 text-blue-700 rounded-lg flex items-center">
                  <AlertCircle className="w-5 h-5 mr-2" />
                  Maximum feedback submissions reached (3/3)
                </div>
              )}

              <div className="flex justify-center space-x-4 mb-4">
                <button
                  onClick={() => setFeedback('up')}
                  className={`p-3 rounded-full transition-colors ${
                    feedback === 'up' ? 'bg-green-100 border border-green-200' :
                    'bg-gray-100 hover:bg-gray-200 border border-gray-200'
                  }`}
                  disabled={isSubmitting}
                >
                  <ThumbsUp className="w-6 h-6 text-green-600" />
                </button>
                <button
                  onClick={() => setFeedback('down')}
                  className={`p-3 rounded-full transition-colors ${
                    feedback === 'down' ? 'bg-red-100 border border-red-200' :
                    'bg-gray-100 hover:bg-gray-200 border border-gray-200'
                  }`}
                  disabled={creditNoteData.feedbackCount>=3 || isSubmitting}
                >
                  <ThumbsDown className="w-6 h-6 text-red-600" />
                </button>
              </div>

              {feedback === 'down' && (
                <div className="space-y-4">
                  <textarea
                    value={feedbackText}
                    onChange={(e) => setFeedbackText(e.target.value)}
                    placeholder="Please specify what needs improvement..."
                    className="w-full px-4 py-3 rounded-lg bg-white border border-gray-300 text-gray-900 placeholder-gray-400 focus:ring-2 focus:ring-red-500 focus:border-transparent transition-all"
                    rows={3}
                    disabled={creditNoteData.feedbackCount >= 3 || isSubmitting}
                  />
                  <div className="flex justify-end space-x-3">
                    <button
                      onClick={() => setFeedback(null)}
                      className="px-4 py-2 text-gray-600 hover:text-gray-900 disabled:opacity-50"
                      disabled={isSubmitting}
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleFeedbackSubmit}
                      disabled={
                        isSubmitting ||
                        !feedbackText.trim() || 
                        creditNoteData.feedbackCount >= 3
                      }
                      className="px-6 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:bg-red-400/50 flex items-center"
                    >
                      {isSubmitting && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                      {isSubmitting ? 'Submitting...' : 'Submit Feedback'}
                    </button>
                  </div>
                </div>
              )}

              {feedback === 'up' && (
                <div className="text-center text-green-600 mt-4">
                  <CheckCircle className="w-8 h-8 mx-auto mb-2" />
                  Thank you for your feedback!
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default CreditNotePage;
