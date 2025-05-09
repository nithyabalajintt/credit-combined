// src/LandingPage.tsx

import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';

const LandingPage = () => {
  const navigate = useNavigate();

  const handleNavigateToHome = () => {
    navigate('/home'); // Navigate to Home Page
  };

  return (
    <div className="min-h-screen flex flex-col font-sans">
      {/* Header */}
      <header className="px-6 py-4 bg-white/90 backdrop-blur-md shadow-sm sticky top-0 z-50">
        <div className="flex flex-col max-w-7xl mx-auto">
          {/* Top Row */}
          <div className="flex justify-between items-center">
            {/* Left side */}
            <div className="flex items-center space-x-2 cursor-pointer" onClick={handleNavigateToHome}>
              <h1 className="text-2xl font-bold text-red-600">RupeeSeetu</h1>
              <span className="text-gray-600 text-sm mt-1">India</span>
            </div>

            {/* Right side */}
            <div className="flex items-center space-x-4">
              <span className="text-gray-700 hover:text-gray-900 cursor-pointer">
                Locations
              </span>
              <button className="border border-gray-400 text-gray-700 hover:text-gray-900 px-4 py-2 rounded-md transition">
                Contact
              </button>
              <button className="bg-red-600 text-white px-4 py-2 rounded-md hover:bg-red-700 transition">
                Welcome Rishitha
              </button>
            </div>
          </div>

          {/* Second Row */}
          <div className="flex space-x-8 mt-4 items-center">
            <span className="text-gray-700 hover:text-gray-900 cursor-pointer">
              Investment Bank
            </span>
            <span className="text-gray-700 hover:text-gray-900 cursor-pointer">
              About us
            </span>
            <span className="text-gray-700 hover:text-gray-900 cursor-pointer">
              Careers
            </span>
            {/* Approve Loan Button */}
            <button
              onClick={handleNavigateToHome}
              className="bg-red-600 text-white px-4 py-2 rounded-md hover:bg-red-700 transition"
            >
              Approve Loans
            </button>
          </div>
        </div>
      </header>

      {/* Main */}
      <main
        className="flex-1 relative bg-cover bg-center"
        style={{
          backgroundImage: `url('https://images.unsplash.com/photo-1601597111158-2fceff292cdc?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80')`
        }}
      >
        {/* Dark overlay */}
        <div className="absolute inset-0 bg-black/30 backdrop-blur-sm"></div>

        {/* Content */}
        <div className="relative z-10 flex items-center justify-start h-full max-w-7xl mx-auto px-6 md:px-8">
          <motion.div
            className="bg-white/90 p-12 max-w-2xl rounded-xl shadow-2xl my-16"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          >
            <div className="border-l-4 border-red-600 pl-6 mb-8">
              <motion.h2
                className="text-4xl md:text-5xl font-light text-gray-900 mb-6"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2, duration: 0.8 }}
              >
                Next-Gen Banking Solutions
              </motion.h2>
              <motion.p
                className="text-gray-700 text-lg md:text-xl leading-relaxed"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4, duration: 0.8 }}
              >
                Empowering your financial journey with secure, AI-driven banking solutions.
                Experience the perfect blend of cutting-edge technology and personalized service.
              </motion.p>
            </div>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.6, duration: 0.8 }}
            >
              <button
                className="bg-red-600 text-white px-8 py-3 text-lg rounded-lg hover:bg-red-700 transition-transform transform hover:scale-105"
              >
                Explore Services
              </button>
            </motion.div>
          </motion.div>
        </div>
      </main>
    </div>
  );
};

export default LandingPage;
