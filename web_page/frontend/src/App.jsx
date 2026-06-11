import { useState, useEffect, useRef } from 'react'
import './components/Header.css'
import './components/CameraFeed.css'
import './components/Footer.css'
import Header from './components/Header'
import CameraFeed from './components/CameraFeed'
import Footer from './components/Footer'

const API_BASE = ''

function App() {
  const [feedState, setFeedState] = useState('idle') // idle | active | error
  const [status, setStatus] = useState({
    connected: false,
    width: 0,
    height: 0,
    resolution: '—',
    device: '—',
    mode: 'unknown',
    frame_count: 0,
  })
  const [streamUrl, setStreamUrl] = useState(null)
  const feedRef = useRef(null)
  const pollRef = useRef(null)

  // Poll backend status while feed is active
  useEffect(() => {
    if (feedState === 'active') {
      const poll = async () => {
        try {
          const res = await fetch(`${API_BASE}/api/status`)
          if (res.ok) {
            const data = await res.json()
            setStatus(data)
          }
        } catch {
          // Backend might be temporarily unreachable
        }
      }
      poll()
      pollRef.current = setInterval(poll, 2000)
      return () => clearInterval(pollRef.current)
    }
  }, [feedState])

  const startFeed = () => {
    // Set stream URL to trigger the <img> to load the MJPEG stream
    setStreamUrl(`${API_BASE}/api/stream?t=${Date.now()}`)
    setFeedState('active')
  }

  const stopFeed = () => {
    setStreamUrl(null)
    setFeedState('idle')
    setStatus(prev => ({
      ...prev,
      connected: false,
      resolution: '—',
      device: '—',
    }))
  }

  const snapshot = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/snapshot`)
      if (res.ok) {
        const blob = await res.blob()
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.download = `visor_${Date.now()}.jpg`
        a.href = url
        a.click()
        URL.revokeObjectURL(url)
      }
    } catch (e) {
      console.error('Snapshot failed:', e)
    }
  }

  const toggleFullscreen = () => {
    const el = feedRef.current
    if (!el) return
    if (document.fullscreenElement) {
      document.exitFullscreen?.()
    } else {
      el.requestFullscreen?.()
    }
  }

  return (
    <div className="app">
      <Header feedActive={feedState === 'active'} />
      <CameraFeed
        ref={feedRef}
        feedState={feedState}
        streamUrl={streamUrl}
        onError={() => setFeedState('error')}
      />
      <Footer
        feedState={feedState}
        status={status}
        onStart={startFeed}
        onStop={stopFeed}
        onSnapshot={snapshot}
        onFullscreen={toggleFullscreen}
      />
    </div>
  )
}

export default App
