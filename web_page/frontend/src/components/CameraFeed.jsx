import { forwardRef, useState, useEffect } from 'react'

const pad = (n) => String(n).padStart(2, '0')

const CameraFeed = forwardRef(function CameraFeed({ feedState, streamUrl, onError }, ref) {
  const [dateStr, setDateStr] = useState('')
  const [timeStr, setTimeStr] = useState('')

  useEffect(() => {
    const tick = () => {
      const now = new Date()
      setDateStr(`${pad(now.getDate())}.${pad(now.getMonth() + 1)}.${now.getFullYear()}`)
      setTimeStr(`${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`)
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  const isActive = feedState === 'active'
  const isError = feedState === 'error'

  return (
    <div className="feed-wrap" id="feed-viewport" ref={ref}>
      {/* No Signal Screen */}
      {!isActive && (
        <div className="no-signal" id="no-signal-screen">
          <svg className="ns-icon" viewBox="0 0 24 24" fill="none" strokeWidth="1" xmlns="http://www.w3.org/2000/svg">
            <rect x="2" y="6" width="14" height="12" rx="1.5" />
            <path d="M22 8.5l-6 3.5 6 3.5V8.5z" />
            <line x1="1" y1="1" x2="23" y2="23" />
          </svg>
          <div className="ns-title" id="ns-title">
            {isError ? 'CONNECTION LOST' : 'NO SIGNAL'}
          </div>
          <div className="ns-sub" id="ns-sub">
            {isError
              ? 'CHECK BACKEND CONNECTION'
              : 'PRESS START FEED TO INITIALIZE'}
          </div>
        </div>
      )}

      {/* MJPEG Stream */}
      {streamUrl && (
        <img
          className="feed-img"
          id="camera-stream"
          src={streamUrl}
          alt="Camera Feed"
          onError={onError}
        />
      )}

      {/* Scanline overlay */}
      <div className="scanlines"></div>

      {/* Corner brackets */}
      <div className="corner tl"></div>
      <div className="corner tr"></div>
      <div className="corner bl"></div>
      <div className="corner br"></div>

      {/* REC badge */}
      {isActive && (
        <div className="rec-wrap" id="rec-badge">
          <div className="rec-badge">
            <div className="rec-blink"></div>
            REC
          </div>
        </div>
      )}

      {/* Camera ID */}
      <div className="cam-id">CAM-01 // PRIMARY</div>

      {/* Timestamp overlay */}
      <div className="ts-wrap">
        <div className="ts-date" id="ts-date">{dateStr}</div>
        <div className="ts-time" id="ts-time">{timeStr}</div>
      </div>
    </div>
  )
})

export default CameraFeed
