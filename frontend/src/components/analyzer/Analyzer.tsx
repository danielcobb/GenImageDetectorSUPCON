import { HERO_IMAGE } from "../../heroImage";
import { useContext, useEffect, useState } from "react";
import {
  Box,
  Button,
  LinearProgress,
  Chip,
  Skeleton,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  styled,
  useTheme,
  Snackbar,
  Alert,
} from "@mui/material";
import {
  CheckCircleOutline,
  WarningAmberRounded,
  AutoAwesome,
  Refresh,
  AddPhotoAlternate,
  CloudUpload,
} from "@mui/icons-material";
import { AppContext, type AppContextType } from "../../contexts/AppContext";
import { AuthContext } from "../../contexts/AuthContext";
import { confidenceToString } from "../../utils";
import { API_URL } from "../../config";

const VisuallyHiddenInput = styled("input")(() => ({
  clip: "rect(0 0 0 0)",
  clipPath: "inset(50%)",
  height: 1,
  overflow: "hidden",
  position: "absolute",
  bottom: 0,
  left: 0,
  whiteSpace: "nowrap",
  width: 1,
}));

const MAX_FILE_SIZE_MB = 15;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;

const MODEL_THRESHOLDS: Record<string, number> = {
  CNNSpot: 80,
  "gid-final": 50,
};
const REAL_CONFIDENCE_THRESHOLD = 55;
const getThreshold = (m?: string) =>
  m
    ? (MODEL_THRESHOLDS[m] ?? REAL_CONFIDENCE_THRESHOLD)
    : REAL_CONFIDENCE_THRESHOLD;

const FEATURES = [
  {
    icon: "⚡",
    label: "Lightning fast",
    desc: "Results in under 10 seconds.",
    color: "rgba(99,102,241,0.08)",
  },
  {
    icon: "🧠",
    label: "Ensemble AI",
    desc: "Multiple models vote for accuracy.",
    color: "rgba(236,72,153,0.08)",
  },
  {
    icon: "📊",
    label: "Full breakdown",
    desc: "See each model's confidence individually.",
    color: "rgba(34,211,238,0.08)",
  },
  {
    icon: "🔐",
    label: "Save history",
    desc: "Sign in to revisit all past analyses.",
    color: "rgba(34,197,94,0.08)",
  },
];

const FUNNY_MESSAGES = [
  "Asking the robots nicely...",
  "Consulting the pixel oracle...",
  "Squinting really hard at your image...",
  "Detecting suspicious vibes...",
  "Running the AI smell test...",
  "Cross-referencing with the Matrix...",
  "Checking if Midjourney left fingerprints...",
  "Almost there, the AI is thinking...",
];

const ConfidenceBar = ({
  confidence,
  modelName,
}: {
  confidence: number;
  modelName?: string;
}) => {
  const isReal = confidence > getThreshold(modelName);
  const [animated, setAnimated] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setAnimated(confidence), 80);
    return () => clearTimeout(t);
  }, [confidence]);
  return (
    <Box sx={{ width: "100%", maxWidth: { xs: "none", sm: 180 } }}>
      <LinearProgress
        variant="determinate"
        value={animated}
        sx={{
          height: 6,
          borderRadius: 3,
          bgcolor: isReal ? "rgba(22,163,74,0.1)" : "rgba(225,29,72,0.1)",
          "& .MuiLinearProgress-bar": {
            bgcolor: isReal ? "#16a34a" : "#e11d48",
            borderRadius: 3,
            transition: "transform 1s cubic-bezier(0.4,0,0.2,1)",
          },
        }}
      />
    </Box>
  );
};

const ConfidenceGauge = ({ confidence }: { confidence: number }) => {
  const theme = useTheme();
  const [animated, setAnimated] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setAnimated(confidence), 100);
    return () => clearTimeout(t);
  }, [confidence]);
  const isReal = confidence >= 50;
  const r = 44;
  const cx = 60;
  const cy = 55;
  const circumference = Math.PI * r;
  const progress = (animated / 100) * circumference;
  const color = isReal ? "#16a34a" : "#e11d48";
  const trackColor =
    theme.palette.mode === "dark"
      ? "rgba(255,255,255,0.12)"
      : "rgba(0,0,0,0.08)";
  return (
    <Box
      sx={{ display: "flex", flexDirection: "column", alignItems: "center" }}
    >
      <svg width="120" height="70" viewBox="0 0 120 70">
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none"
          stroke={trackColor}
          strokeWidth="7"
          strokeLinecap="round"
        />
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none"
          stroke={color}
          strokeWidth="7"
          strokeLinecap="round"
          strokeDasharray={`${progress} ${circumference}`}
          style={{
            transition: "stroke-dasharray 1.5s cubic-bezier(0.4,0,0.2,1)",
          }}
        />
        <text
          x={cx}
          y={cy - 4}
          textAnchor="middle"
          fill={color}
          fontSize="16"
          fontWeight="700"
          fontFamily="inherit"
        >
          {Math.round(animated)}%
        </text>
      </svg>
    </Box>
  );
};

export const Analyzer = () => {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | undefined>(undefined);
  const [loading, setLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState("");
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [funnyMsg, setFunnyMsg] = useState(FUNNY_MESSAGES[0]);
  const [msgIdx, setMsgIdx] = useState(0);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const context = useContext(AppContext);
  const {
    currentResult,
    setCurrentResult,
    history,
    setHistory,
    refreshHistory,
  } = context as AppContextType;
  const authContext = useContext(AuthContext);
  const token = authContext?.token;

  const handleFileSelect = (file: File | null | undefined) => {
    if (!file) return;
    if (file.size > MAX_FILE_SIZE_BYTES) {
      setUploadError(
        `That image is ${(file.size / (1024 * 1024)).toFixed(1)}MB, which is over the ${MAX_FILE_SIZE_MB}MB limit. Please choose a smaller image.`,
      );
      return;
    }
    setUploadError(null);
    setImage(file);
  };

  useEffect(() => {
    if (!loading) return;
    const interval = setInterval(() => {
      setMsgIdx((i) => {
        const next = (i + 1) % FUNNY_MESSAGES.length;
        setFunnyMsg(FUNNY_MESSAGES[next]);
        return next;
      });
    }, 2200);
    return () => clearInterval(interval);
  }, [loading]);

  useEffect(() => {
    const h = () => {
      setImage(null);
      setPreview(undefined);
    };
    window.addEventListener("auth:logout", h);
    window.addEventListener("app:home", h);
    return () => {
      window.removeEventListener("auth:logout", h);
      window.removeEventListener("app:home", h);
    };
  }, []);

  useEffect(() => {
    if (!image) return;
    setCurrentResult(null);
    const reader = new FileReader();
    reader.onloadend = () => {
      if (typeof reader.result === "string") setPreview(reader.result);
    };
    reader.readAsDataURL(image);
  }, [image, setCurrentResult]);

  useEffect(() => {
    if (currentResult) setPreview(currentResult.image);
  }, [currentResult]);

  const reanalyzeImage = async () => {
    if (!currentResult?.image) return;
    const arr = currentResult.image.split(",");
    const mime = arr[0].match(/:(.*?);/)?.[1] || "image/png";
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) u8arr[n] = bstr.charCodeAt(n);
    setImage(
      new File(
        [new Blob([u8arr], { type: mime })],
        currentResult.filename || "reanalyzed.png",
        { type: mime },
      ),
    );
    setTimeout(() => analyzeImage(), 50);
  };

  const analyzeImage = async () => {
    if (!image) return;
    setLoading(true);
    setLoadingProgress(0);
    setCurrentResult(null);
    setMsgIdx(0);
    setFunnyMsg(FUNNY_MESSAGES[0]);
    const formData = new FormData();
    formData.append("file", image);
    const delay = (min: number, max: number) =>
      new Promise((r) => setTimeout(r, Math.random() * (max - min) + min));
    try {
      setLoadingStage("Uploading image...");
      setLoadingProgress(20);
      await delay(400, 800);
      const headers: HeadersInit = {};
      if (token) headers["Authorization"] = `Bearer ${token}`;
      setLoadingStage("Running AI detection models...");
      setLoadingProgress(40);
      await delay(300, 600);
      const res = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        headers,
        body: formData,
      });
      setLoadingStage("Processing results...");
      setLoadingProgress(80);
      await delay(500, 900);
      const data = await res.json();
      const results = Object.entries(data.results).map(
        ([model, confidence]) => ({ model, confidence: confidence as number }),
      );
      const weights = results.map((r) => Math.abs(r.confidence - 50));
      const totalWeight = weights.reduce((s, w) => s + w, 0);
      const aggregateConfidence =
        totalWeight > 0
          ? results.reduce((s, r, i) => s + r.confidence * weights[i], 0) /
            totalWeight
          : 50;
      const newHistoryItem = {
        id: data.analysis_id || undefined,
        image: preview,
        filename: image.name,
        results,
        analysis: {
          model: "Aggregate",
          confidence: Math.round(aggregateConfidence * 10) / 10,
        },
        timestamp: new Date().toISOString(),
      };
      setLoadingStage("Complete!");
      setLoadingProgress(100);
      await delay(400, 600);
      setCurrentResult(newHistoryItem);
      if (data.analysis_id)
        window.history.pushState({}, "", `/analysis/${data.analysis_id}`);
      setLoading(false);
      setLoadingStage("");
      setLoadingProgress(0);
      if (!token) setHistory([newHistoryItem, ...history]);
      else {
        await new Promise((r) => setTimeout(r, 100));
        await refreshHistory();
      }
    } catch (err) {
      console.error(err);
      alert("Error analyzing image");
      setLoading(false);
      setLoadingStage("");
      setLoadingProgress(0);
    }
  };

  const formatTime = (ts: string) => {
    const date = new Date(ts);
    const today = new Date();
    return date.getDate() === today.getDate() &&
      date.getMonth() === today.getMonth() &&
      date.getFullYear() === today.getFullYear()
      ? date.toLocaleTimeString()
      : date.toLocaleString();
  };

  const showLanding = !preview && !loading && !currentResult;

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@700;800&display=swap');
        @keyframes float1{0%,100%{transform:translateY(0) rotate(-1deg)}50%{transform:translateY(-18px) rotate(1deg)}}
        @keyframes float2{0%,100%{transform:translateY(0) rotate(1deg)}50%{transform:translateY(-14px) rotate(-1deg)}}
        @keyframes float3{0%,100%{transform:translateY(0)}50%{transform:translateY(-20px)}}
        @keyframes float4{0%,100%{transform:translateY(0) rotate(1deg)}50%{transform:translateY(-16px) rotate(-2deg)}}
        @keyframes gradText{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
        @keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
        @keyframes dotPulse{0%,100%{transform:scale(1)}50%{transform:scale(1.5)}}
        @keyframes msgFade{0%{opacity:0;transform:translateY(6px)}20%{opacity:1;transform:translateY(0)}80%{opacity:1}100%{opacity:0;transform:translateY(-6px)}}
        @keyframes stepSlide{from{opacity:0;transform:translateX(-12px)}to{opacity:1;transform:translateX(0)}}
      `}</style>

      <Box
        sx={{
          width: "100%",
          maxWidth: 760,
          mx: "auto",
          px: { xs: 2, md: 3 },
          py: 4,
          display: "flex",
          flexDirection: "column",
          gap: 2,
        }}
      >
        {showLanding && (
          <>
            {/* ── HERO: left illustration + right text ── */}
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                gap: { xs: 2, md: 5 },
                flexWrap: "wrap",
                animation: "fadeUp .6s ease forwards",
              }}
            >
              {/* Left: hero image card */}
              <Box
                sx={{
                  flex: "0 0 auto",
                }}
              >
                <Box
                  sx={{
                    width: { xs: 130, md: 175 },
                    borderRadius: 4,
                    overflow: "hidden",
                    boxShadow: "0 20px 60px rgba(99,102,241,0.2)",
                    border: "3px solid #fff",
                    position: "relative",
                  }}
                >
                  <Box
                    component="img"
                    src={HERO_IMAGE}
                    alt="AI vs Real"
                    sx={{
                      width: "100%",
                      height: { xs: 170, md: 210 },
                      objectFit: "cover",
                      display: "block",
                    }}
                    onError={(e: any) => {
                      e.target.style.display = "none";
                      e.target.nextSibling.style.display = "flex";
                    }}
                  />
                  {/* Fallback if image fails */}
                  <Box
                    sx={{
                      display: "none",
                      width: "100%",
                      height: { xs: 170, md: 210 },
                      bgcolor: "#e0e7ff",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 60,
                    }}
                  >
                    🤖
                  </Box>

                  {/* Overlay badge */}
                  <Box
                    sx={{
                      position: "absolute",
                      bottom: 12,
                      left: 12,
                      right: 12,
                      bgcolor: "rgba(255,255,255,0.95)",
                      backdropFilter: "blur(8px)",
                      borderRadius: 2,
                      px: 1.5,
                      py: 1,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                    }}
                  >
                    <Typography
                      sx={{ fontSize: 11, fontWeight: 700, color: "#1a1a2e" }}
                    >
                      AI Generated?
                    </Typography>
                    <Box
                      sx={{
                        fontSize: 10,
                        fontWeight: 800,
                        px: 1,
                        py: 0.3,
                        borderRadius: 1,
                        bgcolor: "#fff1f2",
                        color: "#e11d48",
                        border: "1px solid #fecdd3",
                      }}
                    >
                      ✗ AI GEN
                    </Box>
                  </Box>
                </Box>

                {/* Floating mini badge */}
                <Box
                  sx={{
                    mt: -3,
                    ml: 3,
                    bgcolor: "#fff",
                    borderRadius: 2,
                    px: 2,
                    py: 1,
                    boxShadow: "0 8px 24px rgba(0,0,0,0.1)",
                    border: "1px solid rgba(0,0,0,0.06)",
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 1,
                  }}
                >
                  <Box
                    sx={{
                      width: 8,
                      height: 8,
                      borderRadius: "50%",
                      bgcolor: "#22c55e",
                      boxShadow: "0 0 6px rgba(34,197,94,0.5)",
                    }}
                  />
                  <Typography
                    sx={{ fontSize: 11, fontWeight: 700, color: "#16a34a" }}
                  >
                    94% confidence
                  </Typography>
                </Box>
              </Box>

              {/* Right: hero text */}
              <Box sx={{ flex: 1, minWidth: { xs: "100%", sm: 280 } }}>
                <Typography
                  component="div"
                  sx={{
                    fontFamily: "'Bricolage Grotesque',sans-serif",
                    fontSize: { xs: "2rem", md: "2.8rem" },
                    fontWeight: 800,
                    lineHeight: 1.05,
                    letterSpacing: "normal",
                    mb: 2,
                    color: (t) =>
                      t.palette.mode === "dark" ? "#e8e8f3" : "#1a1a2e",
                  }}
                >
                  Is that photo
                  <br />
                  <Box
                    component="span"
                    sx={{
                      background:
                        "linear-gradient(135deg,#6366f1,#8b5cf6,#ec4899,#f97316)",
                      backgroundSize: "200% auto",
                      WebkitBackgroundClip: "text",
                      WebkitTextFillColor: "transparent",
                      animation: "gradText 5s ease infinite",
                    }}
                  >
                    real or generated?
                  </Box>
                </Typography>
                <Typography
                  sx={{
                    fontSize: 15,
                    color: "#64748b",
                    lineHeight: 1.7,
                    mb: 3,
                  }}
                >
                  Upload any image and our ensemble of deep learning models will
                  tell you if it's authentic or AI-generated — in seconds.
                </Typography>
              </Box>
            </Box>
            {/* ── BIG upload button ── */}
            <Box sx={{ textAlign: "center" }}>
              <Button
                component="label"
                variant="contained"
                size="large"
                startIcon={<CloudUpload sx={{ fontSize: "28px !important" }} />}
                sx={{
                  px: 6,
                  py: 2.5,
                  fontSize: 18,
                  fontWeight: 800,
                  borderRadius: 3,
                  background: "linear-gradient(135deg,#6366f1,#8b5cf6)",
                  boxShadow: "0 12px 40px rgba(99,102,241,0.35)",
                  letterSpacing: "-.01em",
                  "&:hover": {
                    boxShadow: "0 16px 48px rgba(99,102,241,0.5)",
                  },
                }}
              >
                Try With Your Image →
                <VisuallyHiddenInput
                  type="file"
                  accept="image/png,image/jpeg,image/heic,image/heif,.heic,.heif"
                  onChange={(e) => {
                    handleFileSelect(e.target.files?.[0]);
                  }}
                />
              </Button>
              <Typography sx={{ fontSize: 12, color: "#94a3b8", mt: 1.5 }}>
                PNG, JPEG, or HEIC · up to {MAX_FILE_SIZE_MB}MB · Try for free
              </Typography>
            </Box>
            {/* ── Feature cards ── */}
            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit,minmax(150px,1fr))",
                gap: 2,
              }}
            >
              {FEATURES.map((f, i) => (
                <Box
                  key={i}
                  sx={{
                    bgcolor: (t) =>
                      t.palette.mode === "dark" ? "#14141f" : "#fff",
                    borderRadius: 3,
                    p: 3,
                    boxShadow: (t) =>
                      t.palette.mode === "dark"
                        ? "0 4px 20px rgba(0,0,0,0.45)"
                        : "0 4px 20px rgba(0,0,0,0.05)",
                    border: (t) =>
                      t.palette.mode === "dark"
                        ? "1px solid rgba(255,255,255,0.07)"
                        : "1px solid rgba(0,0,0,0.05)",
                    transition: "all .3s",
                    "&:hover": {
                      transform: "translateY(-4px)",
                      boxShadow: "0 12px 32px rgba(99,102,241,0.1)",
                    },
                  }}
                >
                  <Box
                    sx={{
                      width: 44,
                      height: 44,
                      borderRadius: 2,
                      bgcolor: f.color,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 20,
                      mb: 1.75,
                    }}
                  >
                    {f.icon}
                  </Box>
                  <Typography
                    sx={{
                      fontFamily: "'Bricolage Grotesque',sans-serif",
                      fontSize: 14,
                      fontWeight: 700,
                      color: (t) =>
                        t.palette.mode === "dark" ? "#e8e8f3" : "#1a1a2e",
                      mb: 0.75,
                    }}
                  >
                    {f.label}
                  </Typography>
                  <Typography
                    sx={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.6 }}
                  >
                    {f.desc}
                  </Typography>
                </Box>
              ))}
            </Box>
          </>
        )}

        {/* ── PREVIEW ── */}
        {preview && !currentResult && !loading && (
          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 3,
              animation: "fadeUp .4s ease forwards",
            }}
          >
            <Box
              sx={{
                bgcolor: (t) =>
                  t.palette.mode === "dark" ? "#181826" : "#fff",
                border: (t) =>
                  t.palette.mode === "dark"
                    ? "1px solid rgba(255,255,255,0.08)"
                    : "1px solid rgba(0,0,0,0.06)",
                borderRadius: 3,
                p: 3,
                width: "100%",
                textAlign: "center",
                boxShadow: (t) =>
                  t.palette.mode === "dark"
                    ? "0 4px 20px rgba(0,0,0,0.5)"
                    : "0 4px 20px rgba(0,0,0,0.06)",
              }}
            >
              <Typography
                sx={{
                  fontFamily: "'Bricolage Grotesque',sans-serif",
                  fontWeight: 700,
                  fontSize: 14,
                  color: "#94a3b8",
                  mb: 2,
                }}
              >
                Ready to analyze
              </Typography>
              <Box
                component="img"
                src={preview}
                alt="Preview"
                sx={{
                  maxHeight: 280,
                  maxWidth: "100%",
                  objectFit: "contain",
                  borderRadius: 2,
                  boxShadow: "0 8px 24px rgba(0,0,0,0.08)",
                }}
              />
            </Box>
            <Box
              sx={{
                display: "flex",
                flexDirection: { xs: "column", sm: "row" },
                alignItems: "center",
                gap: 2,
              }}
            >
              <Button
                variant="contained"
                size="large"
                onClick={analyzeImage}
                startIcon={<AutoAwesome />}
                sx={{
                  px: { xs: 3, sm: 5 },
                  py: 1.5,
                  fontSize: "1rem",
                  fontWeight: 700,
                  borderRadius: 2,
                }}
              >
                Analyze Image
              </Button>
              <Button
                variant="outlined"
                size="large"
                component="label"
                startIcon={<AddPhotoAlternate />}
                sx={{ borderRadius: 2 }}
              >
                Change Image
                <VisuallyHiddenInput
                  type="file"
                  accept="image/png,image/jpeg,image/heic,image/heif,.heic,.heif"
                  onChange={(e) => {
                    handleFileSelect(e.target.files?.[0]);
                  }}
                />
              </Button>
            </Box>
          </Box>
        )}

        {/* ── LOADING with dancing gif ── */}
        {loading && (
          <Box
            sx={{
              bgcolor: (t) => (t.palette.mode === "dark" ? "#181826" : "#fff"),
              border: (t) =>
                t.palette.mode === "dark"
                  ? "1px solid rgba(255,255,255,0.08)"
                  : "1px solid rgba(0,0,0,0.06)",
              borderRadius: 4,
              p: { xs: 4, md: 6 },
              textAlign: "center",
              boxShadow: (t) =>
                t.palette.mode === "dark"
                  ? "0 8px 40px rgba(0,0,0,0.55)"
                  : "0 8px 40px rgba(0,0,0,0.08)",
              animation: "fadeUp .4s ease forwards",
            }}
          >
            <Typography
              key={msgIdx}
              sx={{
                fontSize: 18,
                fontWeight: 700,
                color: "#6366f1",
                fontFamily: "'Bricolage Grotesque',sans-serif",
                mb: 2,
                animation: "msgFade 2.2s ease forwards",
              }}
            >
              {funnyMsg}
            </Typography>
            <Box sx={{ maxWidth: 400, mx: "auto", mb: 1.5 }}>
              <LinearProgress
                variant="determinate"
                value={loadingProgress}
                sx={{
                  height: 8,
                  borderRadius: 4,
                  bgcolor: "rgba(99,102,241,0.1)",
                  "& .MuiLinearProgress-bar": {
                    background: "linear-gradient(135deg,#6366f1,#ec4899)",
                    borderRadius: 4,
                  },
                }}
              />
            </Box>
            <Typography sx={{ fontSize: 12, color: "#94a3b8", mb: 3 }}>
              {loadingStage}
            </Typography>
            {[1, 2, 3].map((i) => (
              <Skeleton
                key={i}
                variant="rectangular"
                height={48}
                sx={{
                  borderRadius: 2,
                  mb: 1.5,
                  bgcolor: "rgba(99,102,241,0.04)",
                  maxWidth: 500,
                  mx: "auto",
                }}
              />
            ))}
          </Box>
        )}

        {/* ── RESULTS ── */}
        {currentResult && !loading && (
          <Box sx={{ animation: "fadeUp .4s ease forwards" }}>
            <Box
              sx={{
                bgcolor: (t) =>
                  t.palette.mode === "dark" ? "#181826" : "#fff",
                border: (t) =>
                  t.palette.mode === "dark"
                    ? "1px solid rgba(255,255,255,0.08)"
                    : "1px solid rgba(0,0,0,0.06)",
                borderRadius: 3,
                p: 3,
                mb: 2,
                display: "flex",
                alignItems: "center",
                gap: 3,
                flexWrap: "wrap",
                boxShadow: (t) =>
                  t.palette.mode === "dark"
                    ? "0 4px 20px rgba(0,0,0,0.5)"
                    : "0 4px 20px rgba(0,0,0,0.06)",
              }}
            >
              {preview && (
                <Box
                  component="img"
                  src={preview}
                  alt="Preview"
                  sx={{
                    height: 90,
                    maxWidth: 140,
                    objectFit: "cover",
                    borderRadius: 2,
                    border: (t) =>
                      t.palette.mode === "dark"
                        ? "1px solid rgba(255,255,255,0.08)"
                        : "1px solid rgba(0,0,0,0.06)",
                  }}
                />
              )}
              <Box sx={{ flex: 1, minWidth: 0 }}>
                <Typography
                  sx={{
                    fontFamily: "'Bricolage Grotesque',sans-serif",
                    fontWeight: 800,
                    fontSize: 16,
                    color: (t) =>
                      t.palette.mode === "dark" ? "#e8e8f3" : "#1a1a2e",
                    mb: 0.5,
                    overflowWrap: "anywhere",
                    wordBreak: "break-word",
                  }}
                >
                  {currentResult.filename}
                </Typography>
                <Typography sx={{ fontSize: 12, color: "#94a3b8" }}>
                  {formatTime(currentResult.timestamp)}
                </Typography>
              </Box>
              <Box sx={{ display: "flex", gap: 1.5, flexShrink: 0 }}>
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<Refresh />}
                  onClick={reanalyzeImage}
                >
                  Reanalyze
                </Button>
                <Button
                  component="label"
                  variant="outlined"
                  size="small"
                  startIcon={<AddPhotoAlternate />}
                >
                  New Image
                  <VisuallyHiddenInput
                    type="file"
                    accept="image/png,image/jpeg,image/heic,image/heif,.heic,.heif"
                    onChange={(e) => {
                      handleFileSelect(e.target.files?.[0]);
                    }}
                  />
                </Button>
              </Box>
            </Box>
            <Box
              sx={{
                bgcolor: (t) =>
                  t.palette.mode === "dark" ? "#181826" : "#fff",
                border: (t) =>
                  t.palette.mode === "dark"
                    ? "1px solid rgba(255,255,255,0.08)"
                    : "1px solid rgba(0,0,0,0.06)",
                borderRadius: 3,
                overflow: "hidden",
                boxShadow: (t) =>
                  t.palette.mode === "dark"
                    ? "0 4px 20px rgba(0,0,0,0.5)"
                    : "0 4px 20px rgba(0,0,0,0.06)",
              }}
            >
              <Box
                sx={{
                  px: 3,
                  py: 2.5,
                  borderBottom: (t) =>
                    t.palette.mode === "dark"
                      ? "1px solid rgba(255,255,255,0.08)"
                      : "1px solid rgba(0,0,0,0.06)",
                  background: (t) =>
                    t.palette.mode === "dark"
                      ? "linear-gradient(135deg,rgba(99,102,241,0.12),rgba(139,92,246,0.08))"
                      : "linear-gradient(135deg,rgba(99,102,241,0.03),rgba(139,92,246,0.02))",
                }}
              >
                <Typography
                  sx={{
                    fontFamily: "'Bricolage Grotesque',sans-serif",
                    fontWeight: 800,
                    fontSize: 17,
                    color: (t) =>
                      t.palette.mode === "dark" ? "#e8e8f3" : "#1a1a2e",
                  }}
                >
                  Analysis Results
                </Typography>
              </Box>
              <TableContainer
                component={Paper}
                sx={{
                  bgcolor: "transparent",
                  boxShadow: "none",
                  border: "none",
                  display: { xs: "none", sm: "block" },
                }}
              >
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Model</TableCell>
                      <TableCell align="center">Real Confidence</TableCell>
                      <TableCell align="center">Verdict</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {currentResult.results.map((result, idx) => (
                      <TableRow
                        key={idx}
                        sx={{
                          "&:hover": { bgcolor: "rgba(99,102,241,0.02)" },
                          transition: "background .15s",
                        }}
                      >
                        <TableCell>
                          <Box
                            component="code"
                            sx={{
                              fontFamily: "monospace",
                              fontSize: "0.82rem",
                              bgcolor: "rgba(99,102,241,0.06)",
                              border: "1px solid rgba(99,102,241,0.12)",
                              color: "#6366f1",
                              px: 1.2,
                              py: 0.4,
                              borderRadius: 1,
                            }}
                          >
                            {result.model}
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Box
                            sx={{
                              display: "flex",
                              alignItems: "center",
                              gap: 2,
                            }}
                          >
                            <Typography
                              fontWeight={700}
                              fontSize="0.9rem"
                              sx={{
                                minWidth: 44,
                                color:
                                  result.confidence > getThreshold(result.model)
                                    ? "#16a34a"
                                    : "#e11d48",
                              }}
                            >
                              {result.confidence}%
                            </Typography>
                            <ConfidenceBar
                              confidence={result.confidence}
                              modelName={result.model}
                            />
                          </Box>
                        </TableCell>
                        <TableCell align="center">
                          <Chip
                            label={confidenceToString(
                              result.confidence,
                              undefined,
                              undefined,
                              undefined,
                              result.model,
                            )}
                            color={
                              result.confidence > getThreshold(result.model)
                                ? "success"
                                : "error"
                            }
                            variant="outlined"
                            size="small"
                            sx={{ fontWeight: 700, fontSize: "0.72rem" }}
                          />
                        </TableCell>
                      </TableRow>
                    ))}
                    {currentResult.analysis && (
                      <TableRow
                        sx={{
                          bgcolor: "rgba(99,102,241,0.02)",
                          borderTop: "2px solid rgba(99,102,241,0.1)",
                        }}
                      >
                        <TableCell>
                          <Box
                            sx={{
                              display: "flex",
                              alignItems: "center",
                              gap: 1,
                            }}
                          >
                            <Box
                              sx={{
                                width: 8,
                                height: 8,
                                borderRadius: "50%",
                                background:
                                  "linear-gradient(135deg,#6366f1,#8b5cf6)",
                              }}
                            />
                            <Typography
                              fontWeight={800}
                              fontSize="0.9rem"
                              sx={{ fontFamily: "'Bricolage Grotesque',sans-serif" }}
                            >
                              Final Analysis
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <ConfidenceGauge
                            confidence={currentResult.analysis.confidence}
                          />
                        </TableCell>
                        <TableCell align="center">
                          <Chip
                            icon={
                              currentResult.analysis.confidence >= 50 ? (
                                <CheckCircleOutline
                                  sx={{ fontSize: "1rem!important" }}
                                />
                              ) : (
                                <WarningAmberRounded
                                  sx={{ fontSize: "1rem!important" }}
                                />
                              )
                            }
                            label={confidenceToString(
                              currentResult.analysis.confidence,
                              "Likely Real",
                              "Likely AI-generated",
                              undefined,
                              currentResult.analysis.model,
                            )}
                            color={
                              currentResult.analysis.confidence >= 50
                                ? "success"
                                : "error"
                            }
                            size="medium"
                            sx={{
                              fontWeight: 700,
                              fontSize: "0.78rem",
                              px: 0.5,
                            }}
                          />
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </TableContainer>

              {/* Mobile: single-line rows (avoids horizontal scroll) */}
              <Box sx={{ display: { xs: "block", sm: "none" } }}>
                {currentResult.results.map((result, idx) => {
                  const isReal = result.confidence > getThreshold(result.model);
                  return (
                    <Box
                      key={idx}
                      sx={{
                        display: "flex",
                        alignItems: "center",
                        gap: 1,
                        px: 2,
                        py: 1.5,
                        borderBottom: (t) =>
                          t.palette.mode === "dark"
                            ? "1px solid rgba(255,255,255,0.08)"
                            : "1px solid rgba(0,0,0,0.06)",
                      }}
                    >
                      <Box
                        component="code"
                        sx={{
                          fontFamily: "monospace",
                          fontSize: "0.7rem",
                          bgcolor: "rgba(99,102,241,0.06)",
                          border: "1px solid rgba(99,102,241,0.12)",
                          color: "#6366f1",
                          px: 0.5,
                          py: 0.3,
                          borderRadius: 1,
                          flexShrink: 0,
                          width: 68,
                          textAlign: "center",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {result.model}
                      </Box>
                      <Box sx={{ flex: 1, minWidth: 24 }}>
                        <ConfidenceBar
                          confidence={result.confidence}
                          modelName={result.model}
                        />
                      </Box>
                      <Typography
                        fontWeight={700}
                        fontSize="0.78rem"
                        sx={{
                          flexShrink: 0,
                          color: isReal ? "#16a34a" : "#e11d48",
                        }}
                      >
                        {result.confidence}%
                      </Typography>
                      <Chip
                        label={isReal ? "Real" : "AI Gen"}
                        color={isReal ? "success" : "error"}
                        variant="outlined"
                        size="small"
                        sx={{
                          flexShrink: 0,
                          fontWeight: 700,
                          fontSize: "0.62rem",
                          height: 20,
                          "& .MuiChip-label": { px: 0.7 },
                        }}
                      />
                    </Box>
                  );
                })}
                {currentResult.analysis && (
                  <Box
                    sx={{
                      px: 2.5,
                      py: 2.5,
                      bgcolor: "rgba(99,102,241,0.02)",
                      borderTop: "2px solid rgba(99,102,241,0.1)",
                    }}
                  >
                    <Box
                      sx={{
                        display: "flex",
                        alignItems: "center",
                        gap: 1,
                        mb: 1.5,
                      }}
                    >
                      <Box
                        sx={{
                          width: 8,
                          height: 8,
                          borderRadius: "50%",
                          background: "linear-gradient(135deg,#6366f1,#8b5cf6)",
                        }}
                      />
                      <Typography
                        fontWeight={800}
                        fontSize="0.9rem"
                        sx={{ fontFamily: "'Bricolage Grotesque',sans-serif" }}
                      >
                        Final Analysis
                      </Typography>
                    </Box>
                    <Box
                      sx={{
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        gap: 1.5,
                      }}
                    >
                      <ConfidenceGauge
                        confidence={currentResult.analysis.confidence}
                      />
                      <Chip
                        icon={
                          currentResult.analysis.confidence >= 50 ? (
                            <CheckCircleOutline
                              sx={{ fontSize: "1rem!important" }}
                            />
                          ) : (
                            <WarningAmberRounded
                              sx={{ fontSize: "1rem!important" }}
                            />
                          )
                        }
                        label={confidenceToString(
                          currentResult.analysis.confidence,
                          "Likely Real",
                          "Likely AI-generated",
                          undefined,
                          currentResult.analysis.model,
                        )}
                        color={
                          currentResult.analysis.confidence >= 50
                            ? "success"
                            : "error"
                        }
                        size="medium"
                        sx={{
                          fontWeight: 700,
                          fontSize: "0.78rem",
                          px: 0.5,
                        }}
                      />
                    </Box>
                  </Box>
                )}
              </Box>
            </Box>
          </Box>
        )}
      </Box>

      <Snackbar
        open={!!uploadError}
        autoHideDuration={6000}
        onClose={() => setUploadError(null)}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert
          severity="error"
          variant="filled"
          onClose={() => setUploadError(null)}
          sx={{ borderRadius: 2 }}
        >
          {uploadError}
        </Alert>
      </Snackbar>
    </>
  );
};
