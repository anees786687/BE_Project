function Footer({ feedState, status, onStart, onStop, onSnapshot, onFullscreen }) {
  const isActive = feedState === 'active'

  return (
    <footer className="footer" id="app-footer">
      <div className="controls">
        <button
          className="btn go"
          id="start-btn"
          onClick={onStart}
          disabled={isActive}
        >
          <span className="btn-icon tri"></span>
          START FEED
        </button>

        <button
          className="btn"
          id="snapshot-btn"
          onClick={onSnapshot}
          disabled={!isActive}
        >
          <span className="btn-icon cam"></span>
          SNAPSHOT
        </button>

        <button
          className="btn kill"
          id="stop-btn"
          onClick={onStop}
          disabled={!isActive}
        >
          <span className="btn-icon square"></span>
          STOP
        </button>

        <button
          className="btn"
          id="fullscreen-btn"
          onClick={onFullscreen}
        >
          ⛶ FULLSCREEN
        </button>
      </div>

      <div className="meta">
        <div className="meta-item">
          RES <span className="meta-val" id="meta-resolution">
            {status.resolution || '—'}
          </span>
        </div>
        <div className="meta-item">
          DEVICE <span className="meta-val" id="meta-device">
            {status.device || '—'}
          </span>
        </div>
        <div className="meta-item">
          STATUS <span className="meta-val" id="meta-status">
            {isActive ? 'ACTIVE' : feedState === 'error' ? 'ERROR' : 'OFFLINE'}
          </span>
        </div>
      </div>
    </footer>
  )
}

export default Footer
