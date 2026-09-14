import { useState, useRef, useEffect, useMemo, useCallback } from "react";
import Webcam from "react-webcam";
import { Camera, UserPlus, Shield, CheckCircle, XCircle, RefreshCw, ScanFace, Glasses, FlipHorizontal2, ShieldAlert, Users, Eye, Trash2, ImageOff } from "lucide-react";

interface IdentifyMatch {
  box: number[]; // [x_min, y_min, x_max, y_max]
  identified: boolean;
  person_id: string | null;
  name: string;
  score: number;
  avatar_base64?: string | null;
  avatar_url?: string | null;
  match_mode?: string | null;
}

interface Profile {
  person_id: string;
  name: string;
  sample_count: number;
  sample_layers: CaptureLayer[];
  avatar_url?: string | null;
}

interface ProfileSample {
  record_id: string;
  pose_label?: string | null;
  image_url?: string | null;
}

interface ProfileDetail extends Profile {
  samples: ProfileSample[];
}

type CaptureLayer = "plain" | "glasses" | "mask" | "glasses_mask";
type CapturePose = "front" | "left" | "right" | "up" | "down";
type ScanPhase = "idle" | "capturing" | "layer-pause" | "ready" | "submitting" | "done";

interface CaptureStep {
  layer: CaptureLayer;
  pose: CapturePose;
  label: string;
  instruction: string;
}

interface EnrollmentSample {
  image: string;
  poseLabel: string;
  layer: CaptureLayer;
}

interface EnrollmentAnalysis {
  pose: string;
  ready: boolean;
  reason: string;
  brightness?: number;
  sharpness?: number;
}

const API_BASE = "http://localhost:8000/api/v1/faces";
const API_ORIGIN = new URL(API_BASE).origin;
const TARGET_SCAN_PERIOD_MS = 100;
const REQUIRED_CONFIRMATION_FRAMES = 3;
const TRACK_MIN_IOU = 0.5;
const PLAIN_CAPTURE_STEPS: CaptureStep[] = [
  { layer: "plain", pose: "front", label: "plain_front", instruction: "Nhìn thẳng vào camera" },
  { layer: "plain", pose: "left", label: "plain_left", instruction: "Từ từ quay mặt sang trái" },
  { layer: "plain", pose: "right", label: "plain_right", instruction: "Từ từ quay mặt sang phải" },
  { layer: "plain", pose: "up", label: "plain_up", instruction: "Ngẩng mặt nhẹ lên" },
  { layer: "plain", pose: "down", label: "plain_down", instruction: "Cúi mặt nhẹ xuống" },
];
const GLASSES_CAPTURE_STEPS: CaptureStep[] = [
  { layer: "glasses", pose: "front", label: "glasses_front", instruction: "Đeo kính và nhìn thẳng" },
  { layer: "glasses", pose: "left", label: "glasses_left", instruction: "Đeo kính, quay sang trái" },
  { layer: "glasses", pose: "right", label: "glasses_right", instruction: "Đeo kính, quay sang phải" },
];
const MASK_CAPTURE_STEPS: CaptureStep[] = [
  { layer: "mask", pose: "front", label: "mask_front", instruction: "Đeo khẩu trang và nhìn thẳng" },
  { layer: "mask", pose: "left", label: "mask_left", instruction: "Đeo khẩu trang, quay sang trái" },
  { layer: "mask", pose: "right", label: "mask_right", instruction: "Đeo khẩu trang, quay sang phải" },
];
const GLASSES_MASK_CAPTURE_STEPS: CaptureStep[] = [
  { layer: "glasses_mask", pose: "front", label: "glasses_mask_front", instruction: "Đeo kính và khẩu trang, nhìn thẳng" },
  { layer: "glasses_mask", pose: "left", label: "glasses_mask_left", instruction: "Giữ kính và khẩu trang, quay trái" },
  { layer: "glasses_mask", pose: "right", label: "glasses_mask_right", instruction: "Giữ kính và khẩu trang, quay phải" },
];

const CAPTURE_STEPS_BY_LAYER: Record<CaptureLayer, CaptureStep[]> = {
  plain: PLAIN_CAPTURE_STEPS,
  glasses: GLASSES_CAPTURE_STEPS,
  mask: MASK_CAPTURE_STEPS,
  glasses_mask: GLASSES_MASK_CAPTURE_STEPS,
};

const CAPTURE_LAYER_ORDER: CaptureLayer[] = ["plain", "glasses", "mask", "glasses_mask"];

const LAYER_LABELS: Record<CaptureLayer, string> = {
  plain: "Khuôn mặt bình thường",
  glasses: "Khuôn mặt đeo kính",
  mask: "Khuôn mặt đeo khẩu trang",
  glasses_mask: "Kính và khẩu trang",
};

const POSE_LABELS: Record<string, string> = {
  plain_front: "Mặt thường - nhìn thẳng",
  plain_left: "Mặt thường - quay trái",
  plain_right: "Mặt thường - quay phải",
  plain_up: "Mặt thường - ngẩng lên",
  plain_down: "Mặt thường - cúi xuống",
  glasses_front: "Đeo kính - nhìn thẳng",
  glasses_left: "Đeo kính - quay trái",
  glasses_right: "Đeo kính - quay phải",
  mask_front: "Khẩu trang - nhìn thẳng",
  mask_left: "Khẩu trang - quay trái",
  mask_right: "Khẩu trang - quay phải",
  glasses_mask_front: "Kính và khẩu trang - nhìn thẳng",
  glasses_mask_left: "Kính và khẩu trang - quay trái",
  glasses_mask_right: "Kính và khẩu trang - quay phải",
};

const mediaUrl = (path?: string | null) => path ? `${API_ORIGIN}${path}` : "";

const boxIou = (first: number[], second: number[]) => {
  const left = Math.max(first[0], second[0]);
  const top = Math.max(first[1], second[1]);
  const right = Math.min(first[2], second[2]);
  const bottom = Math.min(first[3], second[3]);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const firstArea = Math.max(0, first[2] - first[0]) * Math.max(0, first[3] - first[1]);
  const secondArea = Math.max(0, second[2] - second[0]) * Math.max(0, second[3] - second[1]);
  return intersection / Math.max(firstArea + secondArea - intersection, 1);
};

export default function App() {
  const [activeTab, setActiveTab] = useState<"enroll" | "cctv" | "profiles">("cctv");

  // Tab 1: Enroll States
  const [enrollName, setEnrollName] = useState("");
  const [enrollStatus, setEnrollStatus] = useState<"IDLE" | "LOADING" | "SUCCESS" | "ERROR">("IDLE");
  const [enrollMessage, setEnrollMessage] = useState("");
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [selectedProfileId, setSelectedProfileId] = useState("");
  const [scanPhase, setScanPhase] = useState<ScanPhase>("idle");
  const [captureIndex, setCaptureIndex] = useState(0);
  const [capturedSamples, setCapturedSamples] = useState<EnrollmentSample[]>([]);
  const [enrollmentAnalysis, setEnrollmentAnalysis] = useState<EnrollmentAnalysis | null>(null);
  const [selectedCaptureLayer, setSelectedCaptureLayer] = useState<CaptureLayer>("plain");
  const [isCameraMirrored, setIsCameraMirrored] = useState(true);
  const [submittedSampleCount, setSubmittedSampleCount] = useState(0);
  const [enrollmentPersonId, setEnrollmentPersonId] = useState<string | null>(null);
  const [profileDetail, setProfileDetail] = useState<ProfileDetail | null>(null);
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileActionMessage, setProfileActionMessage] = useState("");
  const captureSteps = useMemo(
    () => CAPTURE_STEPS_BY_LAYER[selectedCaptureLayer],
    [selectedCaptureLayer],
  );
  const selectedProfile = useMemo(
    () => profiles.find((item) => item.person_id === selectedProfileId) || null,
    [profiles, selectedProfileId],
  );

  // Tab 2: CCTV States
  const [isScanning, setIsScanning] = useState(true);
  const [detectedFacesCount, setDetectedFacesCount] = useState(0);
  const [lastMatches, setLastMatches] = useState<IdentifyMatch[]>([]);

  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const recognitionStreaksRef = useRef<Array<{ match: IdentifyMatch; count: number }>>([]);
  const stableCaptureFramesRef = useRef(0);

  const loadProfiles = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/profiles`);
      if (response.ok) {
        const nextProfiles: Profile[] = await response.json();
        setProfiles(nextProfiles);
      }
    } catch {
      // Enrollment remains usable when profile loading is temporarily unavailable.
    }
  }, []);

  const loadProfileDetail = async (personId: string) => {
    setProfileLoading(true);
    setProfileActionMessage("");
    try {
      const response = await fetch(`${API_BASE}/profiles/${personId}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Không thể tải chi tiết hồ sơ.");
      setProfileDetail(data);
    } catch (error) {
      setProfileActionMessage(error instanceof Error ? error.message : "Không thể kết nối đến server AI.");
    } finally {
      setProfileLoading(false);
    }
  };

  const deleteProfileSample = async (sample: ProfileSample) => {
    if (!profileDetail || !window.confirm(`Xóa mẫu “${POSE_LABELS[sample.pose_label || ""] || sample.pose_label || "Chưa gắn nhãn"}”?`)) return;
    setProfileLoading(true);
    try {
      const response = await fetch(`${API_BASE}/profiles/${profileDetail.person_id}/samples/${sample.record_id}`, { method: "DELETE" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Không thể xóa mẫu khuôn mặt.");
      await Promise.all([loadProfiles(), loadProfileDetail(profileDetail.person_id)]);
      setProfileActionMessage(data.message);
    } catch (error) {
      setProfileActionMessage(error instanceof Error ? error.message : "Không thể kết nối đến server AI.");
    } finally {
      setProfileLoading(false);
    }
  };

  const deleteProfile = async (profile: Profile) => {
    if (!window.confirm(`Xóa hồ sơ “${profile.name}” và toàn bộ mẫu khuôn mặt?`)) return;
    setProfileLoading(true);
    try {
      const response = await fetch(`${API_BASE}/profiles/${profile.person_id}`, { method: "DELETE" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Không thể xóa hồ sơ.");
      setProfileDetail(null);
      await loadProfiles();
      setProfileActionMessage(data.message);
    } catch (error) {
      setProfileActionMessage(error instanceof Error ? error.message : "Không thể kết nối đến server AI.");
    } finally {
      setProfileLoading(false);
    }
  };

  const resetAllProfiles = async () => {
    const confirmation = window.prompt("Thao tác này xóa toàn bộ hồ sơ và mẫu ảnh. Nhập RESET để xác nhận.");
    if (confirmation !== "RESET") return;
    setProfileLoading(true);
    try {
      const response = await fetch(`${API_BASE}/profiles?confirm=RESET`, { method: "DELETE" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Không thể reset dữ liệu.");
      setProfileDetail(null);
      setSelectedProfileId("");
      setEnrollName("");
      await loadProfiles();
      setProfileActionMessage(data.message);
    } catch (error) {
      setProfileActionMessage(error instanceof Error ? error.message : "Không thể kết nối đến server AI.");
    } finally {
      setProfileLoading(false);
    }
  };

  useEffect(() => {
    void loadProfiles();
  }, [loadProfiles]);

  // -------------------------------------------------------------
  // Enroll Handlers
  // -------------------------------------------------------------
  const resetEnrollmentScan = () => {
    setScanPhase("idle");
    setCaptureIndex(0);
    setCapturedSamples([]);
    setEnrollmentAnalysis(null);
    setSubmittedSampleCount(0);
    setEnrollmentPersonId(null);
    stableCaptureFramesRef.current = 0;
    const profile = profiles.find((item) => item.person_id === selectedProfileId);
    if (!profile) {
      setSelectedCaptureLayer("plain");
    } else {
      setSelectedCaptureLayer(
        CAPTURE_LAYER_ORDER.find((layer) => !profile.sample_layers.includes(layer)) || "plain",
      );
    }
  };

  const startEnrollmentScan = () => {
    if (!enrollName && !selectedProfileId) {
      setEnrollStatus("ERROR");
      setEnrollMessage("Vui lòng nhập tên hoặc chọn hồ sơ cần bổ sung.");
      return;
    }
    setEnrollStatus("IDLE");
    setEnrollMessage("");
    setCapturedSamples([]);
    setCaptureIndex(0);
    setEnrollmentAnalysis(null);
    setSubmittedSampleCount(0);
    setEnrollmentPersonId(null);
    stableCaptureFramesRef.current = 0;
    setScanPhase("capturing");
  };

  const submitEnrollmentSamples = async () => {
    if (capturedSamples.length !== captureSteps.length) {
      setEnrollStatus("ERROR");
      setEnrollMessage("Phiên quét chưa thu đủ dữ liệu khuôn mặt.");
      return;
    }
    setScanPhase("submitting");
    setEnrollStatus("LOADING");
    try {
      let personId: string | null = enrollmentPersonId || selectedProfileId || null;
      for (let index = submittedSampleCount; index < capturedSamples.length; index += 1) {
        const sample = capturedSamples[index];
        const response = await fetch(`${API_BASE}/enroll`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: enrollName,
            image_base64: sample.image.replace(/^data:image\/[a-z]+;base64,/, ""),
            person_id: personId,
            pose_label: sample.poseLabel,
          }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Không thể lưu bộ dữ liệu khuôn mặt.");
        personId = data.person_id;
        setEnrollmentPersonId(personId);
        setSubmittedSampleCount(index + 1);
      }
      setEnrollStatus("SUCCESS");
      setEnrollMessage(`Đã lưu ${capturedSamples.length} mẫu ${LAYER_LABELS[selectedCaptureLayer].toLowerCase()}.`);
      setScanPhase("done");
      setSelectedProfileId(personId || "");
      void loadProfiles();
    } catch (error) {
      setEnrollStatus("ERROR");
      setEnrollMessage(error instanceof Error ? error.message : "Không thể kết nối đến server AI.");
      setScanPhase("ready");
    }
  };

  useEffect(() => {
    if (activeTab !== "enroll" || scanPhase !== "capturing") return;
    let active = true;
    let timerId: ReturnType<typeof setTimeout> | null = null;
    const step = captureSteps[captureIndex];

    const analyzeFrame = async () => {
      const screenshot = webcamRef.current?.getScreenshot();
      if (!active || !screenshot || !step) {
        if (active) timerId = setTimeout(analyzeFrame, 300);
        return;
      }
      try {
        const response = await fetch(`${API_BASE}/enrollment/analyze`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image_base64: screenshot.replace(/^data:image\/[a-z]+;base64,/, ""),
            expected_pose: step.pose,
            mirrored: isCameraMirrored,
          }),
        });
        if (!response.ok) throw new Error();
        const analysis: EnrollmentAnalysis = await response.json();
        setEnrollmentAnalysis(analysis);
        stableCaptureFramesRef.current = analysis.ready ? stableCaptureFramesRef.current + 1 : 0;
        if (stableCaptureFramesRef.current >= 2) {
          const nextIndex = captureIndex + 1;
          setCapturedSamples((samples) => [...samples, { image: screenshot, poseLabel: step.label, layer: step.layer }]);
          stableCaptureFramesRef.current = 0;
          if (nextIndex >= captureSteps.length) {
            setScanPhase("ready");
          } else {
            setCaptureIndex(nextIndex);
            setScanPhase(captureSteps[nextIndex].layer === step.layer ? "capturing" : "layer-pause");
          }
          return;
        }
      } catch {
        setEnrollmentAnalysis({ pose: "unknown", ready: false, reason: "Không thể phân tích frame hiện tại." });
        stableCaptureFramesRef.current = 0;
      }
      if (active) timerId = setTimeout(analyzeFrame, 300);
    };
    timerId = setTimeout(analyzeFrame, 200);
    return () => {
      active = false;
      if (timerId) clearTimeout(timerId);
    };
  }, [activeTab, captureIndex, captureSteps, isCameraMirrored, scanPhase]);

  // -------------------------------------------------------------
  // CCTV Scanner Simulation Loop
  // -------------------------------------------------------------
  useEffect(() => {
    let active = true;
    let timerId: any = null;

    const performScan = async () => {
      const scanStartedAt = performance.now();
      if (!active || !isScanning || activeTab !== "cctv" || !webcamRef.current) {
        if (active && isScanning && activeTab === "cctv") {
          timerId = setTimeout(performScan, TARGET_SCAN_PERIOD_MS);
        }
        return;
      }

      const screenshot = webcamRef.current.getScreenshot();
      if (!screenshot) {
        if (active) timerId = setTimeout(performScan, TARGET_SCAN_PERIOD_MS);
        return;
      }

      const base64Data = screenshot.replace(/^data:image\/[a-z]+;base64,/, "");

      try {
        const response = await fetch(`${API_BASE}/identify`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: "CCTV_Identify",
            image_base64: base64Data
          }),
        });

        if (response.ok) {
          const data = await response.json();
          const currentMatches: IdentifyMatch[] = data.matches || [];
          const nextStreaks: Array<{ match: IdentifyMatch; count: number }> = [];
          const usedStreakIndexes = new Set<number>();
          const stableMatches = currentMatches.map((match) => {
            if (!match.identified || !match.person_id) return match;
            const previousIndex = recognitionStreaksRef.current.findIndex(
              (item, index) => (
                !usedStreakIndexes.has(index)
                && item.match.person_id === match.person_id
                && boxIou(item.match.box, match.box) >= TRACK_MIN_IOU
              ),
            );
            if (previousIndex >= 0) usedStreakIndexes.add(previousIndex);
            const count = previousIndex >= 0
              ? recognitionStreaksRef.current[previousIndex].count + 1
              : 1;
            nextStreaks.push({ match, count });
            if (count >= REQUIRED_CONFIRMATION_FRAMES) return match;
            return {
              ...match,
              identified: false,
              person_id: null,
              name: "Unknown",
              score: 0,
              avatar_base64: null,
              avatar_url: null,
              match_mode: null,
            };
          });
          recognitionStreaksRef.current = nextStreaks;
          setDetectedFacesCount(data.faces_detected);
          setLastMatches(stableMatches);
        }
      } catch (err) {
        console.error("CCTV Scan network error:", err);
      } finally {
        if (active) {
          const elapsedMs = performance.now() - scanStartedAt;
          timerId = setTimeout(
            performScan,
            Math.max(0, TARGET_SCAN_PERIOD_MS - elapsedMs),
          );
        }
      }
    };

    if (activeTab === "cctv" && isScanning) {
      timerId = setTimeout(performScan, TARGET_SCAN_PERIOD_MS);
    }

    return () => {
      active = false;
      if (timerId) clearTimeout(timerId);
    };
  }, [activeTab, isScanning]);

  // Draw overlay canvas boxes
  useEffect(() => {
    if (activeTab !== "cctv" || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear previous drawing
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    lastMatches.forEach((match) => {
      const [x1, y1, x2, y2] = match.box;
      const width = x2 - x1;
      const height = y2 - y1;

      // Color coding: Green for identified, Red for unknown
      ctx.strokeStyle = match.identified ? "#22c55e" : "#ef4444";
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, width, height);

      // Label background
      ctx.fillStyle = match.identified ? "rgba(34, 197, 94, 0.85)" : "rgba(239, 68, 68, 0.85)";
      const labelText = match.identified ? `${match.name} (${match.score}%)` : "Unknown";
      ctx.font = "bold 14px sans-serif";
      
      const textWidth = ctx.measureText(labelText).width;
      ctx.fillRect(x1, y1 - 25, textWidth + 12, 25);

      // Label text
      ctx.fillStyle = "#ffffff";
      ctx.fillText(labelText, x1 + 6, y1 - 8);
    });
  }, [lastMatches, activeTab]);

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col font-sans">
      {/* Top Navbar */}
      <header className="bg-slate-800 border-b border-slate-700 py-4 px-6 shadow-md flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div className="flex items-center space-x-3">
          <Shield className="w-8 h-8 text-indigo-500" />
          <h1 className="text-xl font-bold tracking-tight text-white">Hệ Thống Nhận Diện Camera CCTV Tối Giản</h1>
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setActiveTab("cctv")}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              activeTab === "cctv"
                ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/30"
                : "bg-slate-700 text-slate-300 hover:bg-slate-600"
            }`}
          >
            <Camera className="w-4 h-4" />
            <span>Giám Sát Camera (Live)</span>
          </button>
          <button
            onClick={() => setActiveTab("enroll")}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              activeTab === "enroll"
                ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/30"
                : "bg-slate-700 text-slate-300 hover:bg-slate-600"
            }`}
          >
            <UserPlus className="w-4 h-4" />
            <span>Thêm Dữ Liệu Gương Mặt</span>
          </button>
          <button
            onClick={() => setActiveTab("profiles")}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              activeTab === "profiles"
                ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/30"
                : "bg-slate-700 text-slate-300 hover:bg-slate-600"
            }`}
          >
            <Users className="w-4 h-4" />
            <span>Quản Lý Hồ Sơ</span>
          </button>
        </div>
      </header>

      {/* Main Body */}
      <main className="flex-1 p-6 max-w-7xl mx-auto w-full flex flex-col justify-center">
        {activeTab === "enroll" ? (
          /* TAB 1: ENROLLMENT */
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-start">
            <div className="md:col-span-8 bg-slate-800 p-5 rounded-lg border border-slate-700 shadow-xl">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-white flex items-center space-x-2">
                  <ScanFace className="w-5 h-5 text-cyan-400" />
                  <span>Quét dữ liệu khuôn mặt 2D</span>
                </h2>
                <span className="text-xs font-mono text-slate-400">{capturedSamples.length}/{captureSteps.length} mẫu</span>
              </div>

              <div className="relative overflow-hidden rounded-lg bg-black aspect-[4/3] w-full border border-slate-600">
                <Webcam
                  audio={false}
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  videoConstraints={{ width: 640, height: 480, facingMode: "user" }}
                  className={`absolute inset-0 w-full h-full object-cover ${isCameraMirrored ? "scale-x-[-1]" : ""}`}
                />
                <button
                  type="button"
                  title="Bật hoặc tắt chế độ soi gương"
                  onClick={() => setIsCameraMirrored((value) => !value)}
                  className="absolute top-3 right-3 z-10 w-10 h-10 bg-slate-950/80 hover:bg-slate-800 border border-slate-600 rounded flex items-center justify-center text-white"
                >
                  <FlipHorizontal2 className="w-5 h-5" />
                </button>
                <div className={`absolute left-1/2 top-1/2 w-[46%] h-[70%] -translate-x-1/2 -translate-y-1/2 rounded-[50%] border-2 ${enrollmentAnalysis?.ready ? "border-emerald-400" : "border-white/70"}`} />
                {scanPhase !== "idle" && scanPhase !== "done" && (
                  <div className="absolute inset-x-4 bottom-4 bg-slate-950/90 border border-slate-700 rounded-lg p-3 text-center">
                    <p className="text-base font-semibold text-white">{captureSteps[captureIndex]?.instruction}</p>
                    <p className={`text-xs mt-1 ${enrollmentAnalysis?.ready ? "text-emerald-400" : "text-slate-300"}`}>
                      {scanPhase === "layer-pause"
                        ? `Chuẩn bị: ${LAYER_LABELS[captureSteps[captureIndex].layer]}`
                        : enrollmentAnalysis?.reason || "Đưa khuôn mặt vào khung oval"}
                    </p>
                  </div>
                )}
              </div>

              <div className="mt-4 h-2 bg-slate-900 rounded overflow-hidden">
                <div className="h-full bg-cyan-500 transition-all" style={{ width: `${capturedSamples.length / captureSteps.length * 100}%` }} />
              </div>
              
              <div className="mt-4 border border-slate-700 bg-slate-900/70 rounded p-3">
                <div className="flex items-center space-x-2 text-xs text-slate-300">
                  {selectedCaptureLayer === "plain" ? <ScanFace className="w-4 h-4" /> : <Glasses className="w-4 h-4" />}
                  <span>{LAYER_LABELS[selectedCaptureLayer]}</span>
                </div>
                <p className="text-xs font-mono text-cyan-400 mt-2">{capturedSamples.length}/{captureSteps.length}</p>
              </div>
            </div>

            <div className="md:col-span-4 bg-slate-800 p-5 rounded-lg border border-slate-700 shadow-xl">
              <h2 className="text-base font-semibold text-white mb-4">Hồ sơ nhận diện</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-1.5">Hồ sơ</label>
                  <select
                    value={selectedProfileId}
                    disabled={scanPhase !== "idle" && scanPhase !== "done"}
                    onChange={(event) => {
                      const personId = event.target.value;
                      setSelectedProfileId(personId);
                      const profile = profiles.find((item) => item.person_id === personId);
                      setEnrollName(profile?.name || "");
                      setSelectedCaptureLayer(
                        profile
                          ? CAPTURE_LAYER_ORDER.find((layer) => !profile.sample_layers.includes(layer)) || "plain"
                          : "plain",
                      );
                    }}
                    className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2.5 text-white disabled:opacity-60"
                  >
                    <option value="">Tạo hồ sơ mới</option>
                    {profiles.map((profile) => <option key={profile.person_id} value={profile.person_id}>{profile.name}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-1.5">Họ và tên</label>
                  <input
                    value={enrollName}
                    onChange={(event) => setEnrollName(event.target.value)}
                    disabled={Boolean(selectedProfileId) || (scanPhase !== "idle" && scanPhase !== "done")}
                    placeholder="Nhập tên người đăng ký"
                    className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2.5 text-white disabled:opacity-60"
                  />
                </div>

                <div className="border-t border-slate-700 pt-4 space-y-3">
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1.5">Loại dữ liệu cần quét</label>
                    <select
                      value={selectedCaptureLayer}
                      disabled={!selectedProfileId || (scanPhase !== "idle" && scanPhase !== "done")}
                      onChange={(event) => setSelectedCaptureLayer(event.target.value as CaptureLayer)}
                      className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2.5 text-white disabled:opacity-60"
                    >
                      <option value="plain">Mặt thường - 5 góc</option>
                      <option value="glasses">Đeo kính - 3 góc</option>
                      <option value="mask">Khẩu trang - 3 góc</option>
                      <option value="glasses_mask">Kính và khẩu trang - 3 góc</option>
                    </select>
                  </div>
                  {selectedProfile && selectedProfile.sample_layers.length > 0 && (
                    <div className="flex flex-wrap gap-1.5">
                      {selectedProfile.sample_layers.map((layer) => (
                        <span key={layer} className="px-2 py-1 border border-emerald-500/30 bg-emerald-500/10 text-emerald-300 rounded text-xs">
                          Đã có: {LAYER_LABELS[layer]}
                        </span>
                      ))}
                    </div>
                  )}
                  <p className="text-xs text-slate-500">
                    Mỗi phiên chỉ quét một loại. Chọn hồ sơ cũ để bổ sung kính hoặc khẩu trang mà không cần quét lại mặt thường.
                  </p>
                </div>

                {scanPhase === "idle" && (
                  <button onClick={startEnrollmentScan} className="w-full bg-cyan-600 hover:bg-cyan-500 text-white py-3 rounded font-semibold flex items-center justify-center space-x-2">
                    <ScanFace className="w-5 h-5" /><span>Bắt đầu quét {LAYER_LABELS[selectedCaptureLayer].toLowerCase()}</span>
                  </button>
                )}
                {scanPhase === "capturing" && (
                  <button disabled className="w-full bg-slate-700 text-slate-300 py-3 rounded font-semibold flex items-center justify-center space-x-2">
                    <RefreshCw className="w-4 h-4 animate-spin" /><span>Đang tự động chụp</span>
                  </button>
                )}
                {scanPhase === "layer-pause" && (
                  <button onClick={() => { setEnrollmentAnalysis(null); setScanPhase("capturing"); }} className="w-full bg-cyan-600 hover:bg-cyan-500 text-white py-3 rounded font-semibold">
                    Tôi đã sẵn sàng
                  </button>
                )}
                {scanPhase === "ready" && (
                  <button onClick={() => void submitEnrollmentSamples()} className="w-full bg-emerald-600 hover:bg-emerald-500 text-white py-3 rounded font-semibold">
                    {submittedSampleCount > 0
                      ? `Tiếp tục lưu ${capturedSamples.length - submittedSampleCount} mẫu còn lại`
                      : `Lưu ${capturedSamples.length} mẫu vào hệ thống`}
                  </button>
                )}
                {scanPhase === "submitting" && (
                  <button disabled className="w-full bg-slate-700 text-slate-300 py-3 rounded font-semibold flex items-center justify-center space-x-2">
                    <RefreshCw className="w-4 h-4 animate-spin" /><span>Đang tạo embedding</span>
                  </button>
                )}
                {scanPhase === "done" && (
                  <button onClick={resetEnrollmentScan} className="w-full bg-slate-700 hover:bg-slate-600 text-white py-3 rounded font-semibold">Tạo phiên quét mới</button>
                )}
                {scanPhase !== "idle" && scanPhase !== "done" && scanPhase !== "submitting" && (
                  <button onClick={resetEnrollmentScan} className="w-full text-sm text-slate-400 hover:text-white py-2">Hủy phiên quét</button>
                )}

                {enrollStatus === "SUCCESS" && (
                  <div className="bg-emerald-500/10 border border-emerald-500/30 p-3 rounded flex items-start space-x-2 text-emerald-400">
                    <CheckCircle className="w-5 h-5 shrink-0" /><p className="text-sm">{enrollMessage}</p>
                  </div>
                )}
                {enrollStatus === "ERROR" && (
                  <div className="bg-rose-500/10 border border-rose-500/30 p-3 rounded flex items-start space-x-2 text-rose-400">
                    <XCircle className="w-5 h-5 shrink-0" /><p className="text-sm">{enrollMessage}</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : activeTab === "profiles" ? (
          <div className="w-full">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-700 pb-4 mb-5">
              <div>
                <h2 className="text-xl font-semibold text-white flex items-center gap-2">
                  <Users className="w-5 h-5 text-cyan-400" />
                  <span>Quản lý hồ sơ nhận diện</span>
                </h2>
                <p className="text-sm text-slate-400 mt-1">{profiles.length} hồ sơ trong hệ thống</p>
              </div>
              <button
                type="button"
                disabled={profileLoading || profiles.length === 0}
                onClick={() => void resetAllProfiles()}
                className="inline-flex items-center justify-center gap-2 px-4 py-2 border border-rose-500/50 text-rose-300 hover:bg-rose-500/10 rounded disabled:opacity-40"
              >
                <Trash2 className="w-4 h-4" />
                <span>Reset toàn bộ dữ liệu</span>
              </button>
            </div>

            {profileActionMessage && (
              <div className="mb-5 border border-slate-700 bg-slate-800 px-4 py-3 rounded text-sm text-slate-200">
                {profileActionMessage}
              </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-[340px_minmax(0,1fr)] gap-6 items-start">
              <section aria-label="Danh sách hồ sơ" className="border border-slate-700 bg-slate-800 rounded overflow-hidden">
                <div className="px-4 py-3 border-b border-slate-700 text-sm font-semibold text-slate-200">Danh sách hồ sơ</div>
                {profiles.length === 0 ? (
                  <div className="px-5 py-12 text-center">
                    <Users className="w-9 h-9 text-slate-600 mx-auto" />
                    <p className="text-sm text-slate-400 mt-3">Chưa có hồ sơ nhận diện.</p>
                  </div>
                ) : (
                  <div className="divide-y divide-slate-700">
                    {profiles.map((profile) => (
                      <button
                        key={profile.person_id}
                        type="button"
                        onClick={() => void loadProfileDetail(profile.person_id)}
                        className={`w-full flex items-center gap-3 p-3 text-left hover:bg-slate-700/60 transition-colors ${profileDetail?.person_id === profile.person_id ? "bg-slate-700" : ""}`}
                      >
                        {profile.avatar_url ? (
                          <img src={mediaUrl(profile.avatar_url)} alt={profile.name} className="w-12 h-12 rounded object-cover border border-slate-600 shrink-0" />
                        ) : (
                          <div className="w-12 h-12 rounded bg-slate-900 border border-slate-600 flex items-center justify-center font-semibold text-cyan-300 shrink-0">
                            {profile.name.slice(0, 1).toUpperCase()}
                          </div>
                        )}
                        <span className="min-w-0 flex-1">
                          <span className="block text-sm font-semibold text-white truncate">{profile.name}</span>
                          <span className="block text-xs text-slate-400 mt-1">{profile.sample_count} mẫu khuôn mặt</span>
                          {profile.sample_layers.length > 0 && (
                            <span className="block text-xs text-cyan-400 mt-1 truncate">
                              {profile.sample_layers.map((layer) => LAYER_LABELS[layer]).join(" · ")}
                            </span>
                          )}
                        </span>
                        <Eye className="w-4 h-4 text-slate-400 shrink-0" />
                      </button>
                    ))}
                  </div>
                )}
              </section>

              <section aria-label="Chi tiết hồ sơ" className="min-w-0">
                {profileLoading && !profileDetail ? (
                  <div className="min-h-64 flex items-center justify-center text-slate-400">
                    <RefreshCw className="w-5 h-5 animate-spin mr-2" /> Đang tải hồ sơ
                  </div>
                ) : !profileDetail ? (
                  <div className="min-h-64 border border-dashed border-slate-700 rounded flex flex-col items-center justify-center text-center px-6">
                    <Eye className="w-9 h-9 text-slate-600" />
                    <p className="text-sm text-slate-400 mt-3">Chọn một hồ sơ để xem và quản lý từng mẫu khuôn mặt.</p>
                  </div>
                ) : (
                  <>
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                      <div>
                        <h3 className="text-lg font-semibold text-white">{profileDetail.name}</h3>
                        <p className="text-xs text-slate-400 mt-1">{profileDetail.samples.length} mẫu đang dùng để nhận diện</p>
                      </div>
                      <button
                        type="button"
                        disabled={profileLoading}
                        onClick={() => void deleteProfile(profileDetail)}
                        className="inline-flex items-center justify-center gap-2 px-3 py-2 bg-rose-600 hover:bg-rose-500 text-white rounded disabled:opacity-50"
                      >
                        <Trash2 className="w-4 h-4" />
                        <span>Xóa hồ sơ</span>
                      </button>
                    </div>

                    {profileDetail.samples.length === 0 ? (
                      <div className="border border-dashed border-slate-700 rounded py-12 text-center text-slate-400">
                        Hồ sơ chưa có mẫu khuôn mặt.
                      </div>
                    ) : (
                      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
                        {profileDetail.samples.map((sample) => (
                          <article key={sample.record_id} className="border border-slate-700 bg-slate-800 rounded overflow-hidden">
                            <div className="aspect-square bg-slate-950 flex items-center justify-center overflow-hidden">
                              {sample.image_url ? (
                                <img src={mediaUrl(sample.image_url)} alt={POSE_LABELS[sample.pose_label || ""] || "Mẫu khuôn mặt"} className="w-full h-full object-cover" loading="lazy" />
                              ) : (
                                <ImageOff className="w-8 h-8 text-slate-600" />
                              )}
                            </div>
                            <div className="p-3 flex items-center gap-2">
                              <p className="text-xs text-slate-200 flex-1 min-w-0 truncate" title={sample.pose_label || "Chưa gắn nhãn"}>
                                {POSE_LABELS[sample.pose_label || ""] || sample.pose_label || "Chưa gắn nhãn"}
                              </p>
                              <button
                                type="button"
                                title="Xóa mẫu khuôn mặt"
                                disabled={profileLoading}
                                onClick={() => void deleteProfileSample(sample)}
                                className="w-8 h-8 flex items-center justify-center text-rose-300 hover:bg-rose-500/10 rounded disabled:opacity-50 shrink-0"
                              >
                                <Trash2 className="w-4 h-4" />
                              </button>
                            </div>
                          </article>
                        ))}
                      </div>
                    )}
                  </>
                )}
              </section>
            </div>
          </div>
        ) : (
          /* TAB 2: CCTV SCANNER */
          <div className="flex flex-col items-center bg-slate-800 p-6 rounded-2xl border border-slate-700 shadow-xl max-w-4xl mx-auto w-full">
            <div className="w-full flex items-center justify-between mb-4 border-b border-slate-700 pb-3">
              <div>
                <h2 className="text-lg font-semibold text-white flex items-center space-x-2">
                  <span className="w-2.5 h-2.5 bg-emerald-500 rounded-full animate-pulse" />
                  <span>Hệ thống giám sát CCTV</span>
                </h2>
                <p className="text-xs text-slate-400 mt-1">Đang phát hiện {detectedFacesCount} gương mặt trong khung hình.</p>
              </div>

              <div className="flex items-center space-x-3">
                <button
                  onClick={() => setIsScanning(!isScanning)}
                  className={`px-4 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
                    isScanning
                      ? "bg-rose-600 text-white hover:bg-rose-500"
                      : "bg-emerald-600 text-white hover:bg-emerald-500"
                  }`}
                >
                  {isScanning ? "Tạm Dừng Quét" : "Bắt Đầu Quét"}
                </button>
              </div>
            </div>

            {/* Webcam & Overlay Canvas Wrapper */}
            <div className="relative overflow-hidden rounded-xl bg-black w-[640px] h-[480px] max-w-full border border-slate-600">
              <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                videoConstraints={{ width: 640, height: 480, facingMode: "user" }}
                className="absolute top-0 left-0 w-full h-full object-cover"
              />
              <canvas
                ref={canvasRef}
                width={640}
                height={480}
                className="absolute top-0 left-0 w-full h-full pointer-events-none"
              />

              {/* Floating Identified Profile Panel on Top-Right Corner */}
              {lastMatches.filter(m => m.identified).length > 0 && (
                <div className="absolute top-4 right-4 z-10 flex flex-col space-y-2 max-w-[240px] transition-all duration-300">
                  {lastMatches.filter(m => m.identified).map((match, idx) => (
                    <div
                      key={idx}
                      className="bg-slate-900/90 backdrop-blur-md border border-emerald-500/40 p-3.5 rounded-xl flex items-center space-x-3 shadow-2xl animate-fade-in"
                    >
                      {match.avatar_url || match.avatar_base64 ? (
                        <img
                          src={match.avatar_url
                            ? mediaUrl(match.avatar_url)
                            : `data:image/jpeg;base64,${match.avatar_base64}`}
                          alt={match.name}
                          className="w-14 h-14 rounded-lg border-2 border-emerald-500 object-cover shrink-0"
                        />
                      ) : (
                        <div className="w-14 h-14 rounded-lg bg-emerald-500/20 border-2 border-emerald-500 flex items-center justify-center text-emerald-400 shrink-0 font-bold text-xl">
                          {match.name ? match.name[0].toUpperCase() : "U"}
                        </div>
                      )}
                      <div className="flex flex-col min-w-0">
                        <span className="text-white font-extrabold text-sm truncate">{match.name}</span>
                        <span className="text-emerald-400 text-xs font-bold mt-1 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-500/20 w-fit">
                          Độ khớp: {match.score}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Detected List Table */}
            <div className="mt-6 w-full max-w-[640px]">
              <h3 className="text-sm font-semibold text-slate-300 mb-2.5">Danh sách nhận diện trong frame cuối:</h3>
              {lastMatches.length === 0 ? (
                <div className="bg-slate-900/60 text-slate-500 text-xs text-center py-4 rounded-lg border border-slate-800">
                  Không phát hiện ai trong camera an ninh
                </div>
              ) : (
                <div className="space-y-2">
                  {lastMatches.map((match, idx) => (
                    <div
                      key={idx}
                      className={`flex justify-between items-center p-3 rounded-xl border text-sm ${
                        match.identified
                          ? "bg-emerald-500/5 border-emerald-500/20 text-emerald-400"
                          : "bg-rose-500/5 border-rose-500/20 text-rose-400"
                      }`}
                    >
                      <div className="flex items-center space-x-3">
                        {match.identified && (match.avatar_url || match.avatar_base64) && (
                          <img
                            src={match.avatar_url
                              ? mediaUrl(match.avatar_url)
                              : `data:image/jpeg;base64,${match.avatar_base64}`}
                            alt={match.name}
                            className="w-10 h-10 rounded-full border border-emerald-500/40 object-cover shrink-0"
                          />
                        )}
                        <div className="flex items-center space-x-2">
                          <span className={`w-2 h-2 rounded-full ${match.identified ? "bg-emerald-500" : "bg-rose-500"}`} />
                          <span className="font-semibold text-slate-200">
                            {match.identified ? match.name : "Người Lạ (Unknown)"}
                          </span>
                        </div>
                      </div>
                      {match.identified && (
                        <span className="text-xs font-mono bg-slate-900/80 px-2.5 py-1 rounded-md text-slate-400 border border-slate-800">
                          Match: {match.score}%
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
