import Hero from "../components/sections/Hero";
import TrustSection from "../components/sections/TrustSection";
import Features from "../components/sections/Features";
import HowItWorks from "../components/sections/HowItWorks";
import ProjectJourney from "../components/sections/ProjectJourney";
import ModelPerformance from "../components/sections/ModelPerformance";
import ScreenshotsCarousel from "../components/sections/ScreenshotsCarousel";
import TechStack from "../components/sections/TechStack";
import About from "../components/sections/About";
import Roadmap from "../components/sections/Roadmap";
import CallToAction from "../components/sections/CallToAction";
import Footer from "../components/sections/Footer";


export default function Page() {
  return (
    <main className="min-h-screen bg-[#070812] text-white">
      <Hero />
      <TrustSection />
      <Features />
      <HowItWorks />
      <ProjectJourney />
      <ModelPerformance />
      <ScreenshotsCarousel />
      <TechStack />
      <About />
      <Roadmap />
      <CallToAction />
      <Footer />
    </main>
  );
}

