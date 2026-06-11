import { useState, useEffect } from 'react'

const pad = (n) => String(n).padStart(2, '0')

function Header({ feedActive }) {
  const [time, setTime] = useState('')

  useEffect(() => {
    const tick = () => {
      const now = new Date()
      setTime(`${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`)
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  return (
    <header className="header" id="app-header">
      <div className="logo">
        VISOR <span className="logo-sub">CAMERA FEED</span>
      </div>
      <div className="hdr-right">
        <div className={`live-pill ${feedActive ? 'on' : ''}`} id="live-pill">
          <div className="live-dot"></div>
          LIVE
        </div>
        <div className="clock" id="header-clock">{time || '--:--:--'}</div>
      </div>
    </header>
  )
}

export default Header
