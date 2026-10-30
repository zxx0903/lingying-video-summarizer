import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App'
import { t } from './i18n'

const manifestProject = {
  project_id: 'proj-real',
  source_video_title: 'Manifest Project',
  source_video_path: 'source.mp4',
  source_video_url: '/api/projects/proj-real/media/source',
  source_video_duration: 180,
  updated_at: '2026-04-20T00:00:00Z',
  clips: [
    {
      clip_id: 'clip-real-1',
      title: 'Loaded Clip',
      video_part: 'part01',
      start_time: '00:00:05',
      end_time: '00:00:20',
      absolute_start_time: '00:00:05',
      absolute_end_time: '00:00:20',
      absolute_time_range: '00:00:05 - 00:00:20',
      speed: 1,
      asset_registry: { current_composed_clip: 'clips_post_processed/clip-real-1.mp4' },
      subtitle_segments: [
        { start_time: '00:00:00,000', end_time: '00:00:02,500', text: 'Loaded subtitle' },
        { start_time: '00:00:02,500', end_time: '00:00:05,000', text: 'Second loaded line' },
      ],
      translated_subtitle_segments: [
        { start_time: '00:00:00,000', end_time: '00:00:02,500', text: '已加载字幕' },
        { start_time: '00:00:02,500', end_time: '00:00:05,000', text: '第二行译文' },
      ],
      subtitle_recipe: { override_text: 'Loaded subtitle' },
      effective_subtitle_text: 'Loaded subtitle\nSecond loaded line',
      translated_subtitle_text: '已加载字幕\n第二行译文',
      has_translated_subtitles: true,
      has_manual_subtitle_override: true,
      cover_recipe: { text: 'Loaded cover' },
      current_composed_clip_url: '/api/projects/proj-real/media/clips_post_processed/clip-real-1.mp4',
      horizontal_cover_url: '/api/projects/proj-real/media/covers/cover-clip-real-1-horizontal.jpg',
      vertical_cover_url: '/api/projects/proj-real/media/covers/cover-clip-real-1-vertical.jpg',
      recovery: {},
      metadata: {},
    },
    {
      clip_id: 'clip-real-2',
      title: 'Second Clip',
      video_part: 'part01',
      start_time: '00:00:25',
      end_time: '00:00:40',
      absolute_start_time: '00:00:25',
      absolute_end_time: '00:00:40',
      absolute_time_range: '00:00:25 - 00:00:40',
      speed: 1,
      asset_registry: { current_composed_clip: 'clips_post_processed/clip-real-2.mp4' },
      subtitle_segments: [
        { start_time: '00:00:00,000', end_time: '00:00:03,000', text: 'Second subtitle' },
      ],
      translated_subtitle_segments: [],
      subtitle_recipe: { override_text: 'Second subtitle' },
      effective_subtitle_text: 'Second subtitle',
      translated_subtitle_text: '',
      has_translated_subtitles: false,
      has_manual_subtitle_override: false,
      cover_recipe: { text: 'Second cover' },
      current_composed_clip_url: '/api/projects/proj-real/media/clips_post_processed/clip-real-2.mp4',
      horizontal_cover_url: '/api/projects/proj-real/media/covers/cover-clip-real-2-horizontal.jpg',
      vertical_cover_url: '/api/projects/proj-real/media/covers/cover-clip-real-2-vertical.jpg',
      recovery: {},
      metadata: {},
    },
  ],
}

describe('OpenClip editor shell', () => {
  const playSpy = vi.fn().mockResolvedValue(undefined)
  const pauseSpy = vi.fn()
  const localStorageState = new Map<string, string>()
  const localStorageMock = {
    getItem: vi.fn((key: string) => localStorageState.get(key) ?? null),
    setItem: vi.fn((key: string, value: string) => {
      localStorageState.set(key, value)
    }),
    removeItem: vi.fn((key: string) => {
      localStorageState.delete(key)
    }),
    clear: vi.fn(() => {
      localStorageState.clear()
    }),
  }

  Object.defineProperty(HTMLMediaElement.prototype, 'play', {
    configurable: true,
    writable: true,
    value: playSpy,
  })
  Object.defineProperty(HTMLMediaElement.prototype, 'pause', {
    configurable: true,
    writable: true,
    value: pauseSpy,
  })
  Object.defineProperty(window, 'localStorage', {
    configurable: true,
    writable: true,
    value: localStorageMock,
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    playSpy.mockClear()
    pauseSpy.mockClear()
    localStorageMock.clear()
    localStorageMock.getItem.mockClear()
    localStorageMock.setItem.mockClear()
    localStorageMock.removeItem.mockClear()
    localStorageMock.clear.mockClear()
    document.documentElement.style.colorScheme = ''
  })

  const zh = (key: Parameters<typeof t>[1], values?: Parameters<typeof t>[2]) => t('zh', key, values)
  const en = (key: Parameters<typeof t>[1], values?: Parameters<typeof t>[2]) => t('en', key, values)
  const getTimelinePreviewVideo = () => document.querySelector('.timeline-shell__preview video') as HTMLVideoElement | null

  it('shows a safe fallback when the editor service cannot load', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('Project demo-openclip-project not found yet; showing demo shell fallback.')))

    render(<App />)

    await screen.findAllByText(/showing demo shell fallback/i)
    expect(screen.getByRole('heading', { name: zh('editorUnavailable') })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: zh('retryLoad') })).toBeInTheDocument()
  })

  it('loads a manifest-backed project when the editor service responds', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => manifestProject }))

    render(<App />)

    expect(await screen.findByRole('heading', { name: 'Loaded Clip' })).toBeInTheDocument()
    expect(screen.getByText(zh('loadedManifestProjectStatus', { projectName: 'Manifest Project' }))).toBeInTheDocument()
    expect(screen.getByDisplayValue('Loaded subtitle')).toBeInTheDocument()
    expect(screen.getByDisplayValue('Second loaded line')).toBeInTheDocument()
    expect(screen.getByDisplayValue('已加载字幕')).toBeInTheDocument()
    expect(screen.getByDisplayValue('第二行译文')).toBeInTheDocument()
    expect(screen.getAllByText('00:00:00,000 → 00:00:02,500')).toHaveLength(1)
    expect(screen.getByRole('article', { name: zh('postProcessedPreview') })).toBeInTheDocument()
    expect(screen.getByRole('article', { name: zh('horizontalCoverPreview') })).toBeInTheDocument()
    expect(screen.getByRole('article', { name: zh('verticalCoverPreview') })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: zh('openDiagnosticsDrawer') })).toBeInTheDocument()
    await waitFor(() => expect(getTimelinePreviewVideo()?.currentTime).toBe(5))
    expect(document.querySelector('.shell[data-theme="light"]')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: zh('switchToDarkTheme') })).toBeInTheDocument()
  })

  it('opens and closes diagnostics in a right-side drawer', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => manifestProject }))
    const user = userEvent.setup()

    render(<App />)

    await screen.findByRole('heading', { name: 'Loaded Clip' })
    const diagnosticsButton = screen.getByRole('button', { name: zh('openDiagnosticsDrawer') })
    const drawer = screen.getByRole('complementary', { name: zh('diagnosticsDrawer') })

    expect(drawer.className).not.toContain('diagnostics-drawer--open')

    await user.click(diagnosticsButton)
    expect(drawer.className).toContain('diagnostics-drawer--open')
    expect(screen.getByRole('heading', { name: zh('editorDiagnostics') })).toBeInTheDocument()
    expect(screen.getByText(zh('recentActivity'))).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: zh('closeDiagnosticsDrawer') }))
    expect(drawer.className).not.toContain('diagnostics-drawer--open')
  })

  it('defaults to Chinese and can switch language while persisting the preference', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => manifestProject }))
    const user = userEvent.setup()

    render(<App />)

    await screen.findByRole('heading', { name: 'Loaded Clip' })
    expect(screen.getByRole('heading', { name: zh('subtitleEditor') })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: zh('switchToDarkTheme') })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'EN' }))

    expect(screen.getByRole('heading', { name: en('subtitleEditor') })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: en('switchToDarkTheme') })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: en('openDiagnosticsDrawer') })).toBeInTheDocument()
    expect(localStorageMock.getItem('openclip-editor-language')).toBe('en')

    await user.click(screen.getByRole('button', { name: '中文' }))

    expect(screen.getByRole('heading', { name: zh('subtitleEditor') })).toBeInTheDocument()
    expect(localStorageMock.getItem('openclip-editor-language')).toBe('zh')
  })

  it('keeps unsaved subtitle edits when switching language', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => manifestProject })
    vi.stubGlobal('fetch', fetchMock)

    render(<App />)

    await screen.findByRole('heading', { name: 'Loaded Clip' })
    const subtitleField = screen.getByLabelText(zh('subtitleCueTextareaLabel', { index: 1 }))

    await user.clear(subtitleField)
    await user.type(subtitleField, 'Unsaved bilingual draft')
    expect(screen.getByDisplayValue('Unsaved bilingual draft')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'EN' }))

    expect(screen.getByRole('heading', { name: en('subtitleEditor') })).toBeInTheDocument()
    expect(screen.getByDisplayValue('Unsaved bilingual draft')).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('saves translated subtitle edits before queueing subtitle rerender', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.fn()
    fetchMock
      .mockResolvedValueOnce({ ok: true, json: async () => manifestProject })
      .mockImplementationOnce(async (_input: RequestInfo | URL, init?: RequestInit) => {
        expect(JSON.parse(String(init?.body))).toMatchObject({
          subtitle_segments: [
            { start_time: '00:00:00,000', end_time: '00:00:02,500', text: '已更新译文' },
            { start_time: '00:00:02,500', end_time: '00:00:05,000', text: '第二行译文' },
          ],
        })
        return { ok: true, json: async () => ({}) }
      })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ job_id: 'job-translated-1' }) })
      .mockImplementation((input: RequestInfo | URL) => {
        const url = String(input)
        if (url === '/api/jobs/job-translated-1') {
          return new Promise(() => {})
        }
        throw new Error(`Unexpected fetch in test: ${url}`)
      })
    vi.stubGlobal('fetch', fetchMock)

    render(<App />)

    await screen.findByRole('heading', { name: 'Loaded Clip' })
    const translatedField = screen.getByLabelText(zh('translatedSubtitleCueTextareaLabel', { index: 1 }))
    await user.clear(translatedField)
    await user.type(translatedField, '已更新译文')
    await user.click(screen.getByRole('button', { name: zh('queueRerender', { operation: zh('operationSubtitle') }) }))

    expect(screen.getByRole('button', { name: zh('rerenderInProgress', { operation: zh('operationSubtitle') }) })).toBeDisabled()
  })

  it('toggles to dark theme and persists the preference', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => manifestProject }))
    const user = userEvent.setup()

    render(<App />)

    await screen.findByRole('heading', { name: 'Loaded Clip' })
    await user.click(screen.getByRole('button', { name: zh('switchToDarkTheme') }))

    expect(document.querySelector('.shell[data-theme="dark"]')).toBeInTheDocument()
    expect(localStorageMock.getItem('openclip-editor-theme')).toBe('dark')
    expect(document.documentElement.style.colorScheme).toBe('dark')
    expect(screen.getByRole('button', { name: zh('switchToLightTheme') })).toBeInTheDocument()
  })

  it('uses the saved light theme preference on first render', async () => {
    localStorageMock.setItem('openclip-editor-theme', 'light')
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => manifestProject }))

    render(<App />)

    await screen.findByRole('heading', { name: 'Loaded Clip' })
    expect(document.querySelector('.shell[data-theme="light"]')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: zh('switchToDarkTheme') })).toBeInTheDocument()
  })

  it('keeps timeline above subtitle and cover sections and shows the rendered subtitle preview', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => manifestProject }))

    render(<App />)

    await screen.findByRole('heading', { name: 'Loaded Clip' })
    const timelineHeading = screen.getByRole('heading', { name: zh('timelinePreview') })
    const subtitleHeading = screen.getByRole('heading', { name: zh('subtitleEditor') })
    const coverHeading = screen.getByRole('heading', { name: zh('coverTitleEditor') })
    const previewPanel = screen.getByRole('article', { name: zh('postProcessedPreview') })
    const previewVideo = previewPanel.querySelector('video')

    expect(timelineHeading.compareDocumentPosition(subtitleHeading) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(subtitleHeading.compareDocumentPosition(coverHeading) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(previewPanel).toHaveTextContent(zh('postProcessedPreviewDescription'))
    expect(previewVideo?.getAttribute('src')).toBe('/api/projects/proj-real/media/clips_post_processed/clip-real-1.mp4?v=2026-04-20T00%3A00%3A00Z')
  })

  it('selecting another clip auto-seeks preview without autoplay even when the source URL is unchanged', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => manifestProject }))
    const user = userEvent.setup()

    render(<App />)

    await screen.findByRole('heading', { name: 'Loaded Clip' })
    const video = document.querySelector('video') as HTMLVideoElement
    expect(video).toBeTruthy()

    await user.click(screen.getByRole('button', { name: /Second Clip/i }))

    await waitFor(() => expect(screen.getByRole('heading', { name: 'Second Clip' })).toBeInTheDocument())
    expect(video.currentTime).toBe(25)
    expect(pauseSpy).toHaveBeenCalled()
    expect(playSpy).not.toHaveBeenCalled()
  })

  it('changing boundary controls updates the timeline preview immediately', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => manifestProject }))

    render(<App />)

    await screen.findByRole('heading', { name: 'Loaded Clip' })
    fireEvent.change(screen.getByLabelText(zh('clipStart')), { target: { value: '6' } })

    await waitFor(() => expect(getTimelinePreviewVideo()?.currentTime).toBe(6))
    expect(screen.getByLabelText(zh('clipStart'))).toHaveValue('6')
  })

  it('changing playback speed updates output duration and reset restores the saved speed', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => manifestProject }))
    const user = userEvent.setup()

    render(<App />)

    await screen.findByRole('heading', { name: 'Loaded Clip' })
    const speedSelect = screen.getByLabelText(zh('playbackSpeed'))

    expect(speedSelect).toHaveValue('1')
    expect(screen.getByText('00:15.0')).toBeInTheDocument()

    fireEvent.change(speedSelect, { target: { value: '2' } })

    expect(speedSelect).toHaveValue('2')
    expect(screen.getByText('00:07.5')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: zh('resetClipDraft') }))

    expect(screen.getByLabelText(zh('playbackSpeed'))).toHaveValue('1')
    expect(screen.getByText('00:15.0')).toBeInTheDocument()
  })

  it('resetting the clip draft realigns the timeline preview to the saved clip bounds', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => manifestProject }))
    const user = userEvent.setup()

    render(<App />)

    await screen.findByRole('heading', { name: 'Loaded Clip' })
    fireEvent.change(screen.getByLabelText(zh('clipStart')), { target: { value: '6' } })
    await waitFor(() => expect(getTimelinePreviewVideo()?.currentTime).toBe(6))

    await user.click(screen.getByRole('button', { name: zh('resetClipDraft') }))

    await waitFor(() => expect(getTimelinePreviewVideo()?.currentTime).toBe(5))
    expect(screen.getByLabelText(zh('clipStart'))).toHaveValue('5')
  })

  it('shows inline boundary rerender feedback while a boundary rerender is in progress', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        ...manifestProject,
        clips: [{
          ...manifestProject.clips[0],
          recovery: { pending_job_id: 'job-boundary-1', pending_operation: 'boundary' },
        }],
      }),
    }))

    render(<App />)

    await screen.findByRole('heading', { name: 'Loaded Clip' })
    expect(screen.getByRole('button', { name: zh('rerenderInProgress', { operation: zh('operationBoundary') }) })).toBeDisabled()
    expect(screen.getByText(zh('boundaryRerenderStarted'))).toBeInTheDocument()
  })

  it('renders rerender status copy in English after switching locale', async () => {
    localStorageMock.setItem('openclip-editor-language', 'en')
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        ...manifestProject,
        clips: [{
          ...manifestProject.clips[0],
          recovery: { pending_job_id: 'job-boundary-en', pending_operation: 'boundary' },
        }],
      }),
    }))

    render(<App />)

    await screen.findByRole('heading', { name: 'Loaded Clip' })
    expect(screen.getByRole('button', { name: en('rerenderInProgress', { operation: en('operationBoundary') }) })).toBeDisabled()
    expect(screen.getByText(en('boundaryRerenderStarted'))).toBeInTheDocument()
  })

  it('shows rerender feedback only in the matching section', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.fn()
    fetchMock
      .mockResolvedValueOnce({ ok: true, json: async () => manifestProject })
      .mockResolvedValueOnce({ ok: true, json: async () => ({}) })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ job_id: 'job-subtitle-1' }) })
      .mockImplementation((input: RequestInfo | URL) => {
        const url = String(input)
        if (url === '/api/jobs/job-subtitle-1') {
          return new Promise(() => {})
        }
        throw new Error(`Unexpected fetch in test: ${url}`)
      })
    vi.stubGlobal('fetch', fetchMock)

    render(<App />)

    await screen.findByRole('heading', { name: 'Loaded Clip' })
    const subtitleField = screen.getByLabelText(zh('subtitleCueTextareaLabel', { index: 1 }))
    await user.clear(subtitleField)
    await user.type(subtitleField, 'Queued subtitle change.')
    await user.click(screen.getByRole('button', { name: zh('queueRerender', { operation: zh('operationSubtitle') }) }))

    expect(screen.getByRole('button', { name: zh('rerenderInProgress', { operation: zh('operationSubtitle') }) })).toBeDisabled()
    expect(screen.getByText(zh('subtitleRerenderStarted'))).toBeInTheDocument()
    expect(screen.getByText(zh('queuedRerenderStatus', { operation: zh('operationSubtitle'), title: 'Loaded Clip' }))).toBeInTheDocument()
    expect(screen.queryByText(zh('boundaryRerenderStarted'))).toBeNull()
    expect(screen.queryByText(zh('coverRerenderStarted'))).toBeNull()
  })

  it('does not show a stale subtitle warning when boundaries move before rerender', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => manifestProject }))

    render(<App />)

    await screen.findByRole('heading', { name: 'Loaded Clip' })
    const startSlider = screen.getByLabelText(zh('clipStart'))
    fireEvent.change(startSlider, { target: { value: '6' } })

    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('reloads regenerated subtitle cues immediately after queueing a boundary rerender', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.fn()
    fetchMock
      .mockResolvedValueOnce({ ok: true, json: async () => manifestProject })
      .mockImplementationOnce(async (_input: RequestInfo | URL, init?: RequestInit) => {
        expect(JSON.parse(String(init?.body))).toMatchObject({
          start_time: '00:00:06.000',
          end_time: '00:00:20.000',
          speed: 2,
        })
        return { ok: true, json: async () => ({}) }
      })
      .mockResolvedValueOnce({ ok: true, json: async () => ({}) })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ job_id: 'job-boundary-refresh' }) })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          ...manifestProject,
          clips: [{
            ...manifestProject.clips[0],
            start_time: '00:00:06',
            absolute_start_time: '00:00:06',
            absolute_time_range: '00:00:06 - 00:00:20',
            speed: 2,
            subtitle_segments: [
              { start_time: '00:00:00,000', end_time: '00:00:01,200', text: 'Boundary-synced cue' },
              { start_time: '00:00:01,200', end_time: '00:00:02,400', text: 'Follow-up synced cue' },
            ],
            effective_subtitle_text: 'Boundary-synced cue\nFollow-up synced cue',
            subtitle_recipe: {},
            has_manual_subtitle_override: false,
            recovery: { pending_job_id: 'job-boundary-refresh', pending_operation: 'boundary' },
          }],
        }),
      })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ status: 'completed' }) })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          ...manifestProject,
          clips: [{
            ...manifestProject.clips[0],
            start_time: '00:00:06',
            absolute_start_time: '00:00:06',
            absolute_time_range: '00:00:06 - 00:00:20',
            speed: 2,
            subtitle_segments: [
              { start_time: '00:00:00,000', end_time: '00:00:01,200', text: 'Boundary-synced cue' },
              { start_time: '00:00:01,200', end_time: '00:00:02,400', text: 'Follow-up synced cue' },
            ],
            effective_subtitle_text: 'Boundary-synced cue\nFollow-up synced cue',
            subtitle_recipe: {},
            has_manual_subtitle_override: false,
            recovery: {},
          }],
        }),
      })
    vi.stubGlobal('fetch', fetchMock)

    render(<App />)

    await screen.findByRole('heading', { name: 'Loaded Clip' })
    const subtitleField = screen.getByLabelText(zh('subtitleCueTextareaLabel', { index: 1 }))
    await user.clear(subtitleField)
    await user.type(subtitleField, 'Manual override before confirming boundary')
    fireEvent.change(screen.getByLabelText(zh('clipStart')), { target: { value: '6' } })
    fireEvent.change(screen.getByLabelText(zh('playbackSpeed')), { target: { value: '2' } })
    await user.click(screen.getByRole('button', { name: zh('queueRerender', { operation: zh('operationBoundary') }) }))

    await waitFor(() => expect(screen.getByDisplayValue('Boundary-synced cue')).toBeInTheDocument())
    expect(screen.getByDisplayValue('Follow-up synced cue')).toBeInTheDocument()
    expect(screen.queryByDisplayValue('Manual override before confirming boundary')).toBeNull()
    expect(screen.getByLabelText(zh('playbackSpeed'))).toHaveValue('2')
    await waitFor(() => expect(getTimelinePreviewVideo()?.currentTime).toBe(6))
  })

  it('waits for project reconciliation after a completed boundary job before clearing the in-progress UI', async () => {
    const user = userEvent.setup()
    let projectLoadCount = 0
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url === '/api/projects/' && (!init?.method || init.method === 'GET')) {
        projectLoadCount += 1
        if (projectLoadCount === 1) {
          return { ok: true, json: async () => manifestProject }
        }
        if (projectLoadCount === 2) {
          return {
            ok: true,
            json: async () => ({
              ...manifestProject,
              clips: [{
                ...manifestProject.clips[0],
                subtitle_segments: [{ start_time: '00:00:00,000', end_time: '00:00:03,000', text: 'Boundary-synced subtitle text' }],
                effective_subtitle_text: 'Boundary-synced subtitle text',
                subtitle_recipe: {},
                has_manual_subtitle_override: false,
                recovery: { pending_job_id: 'job-boundary-2', pending_operation: 'boundary' },
              }],
            }),
          }
        }
        return {
          ok: true,
          json: async () => ({
            ...manifestProject,
            clips: [{
              ...manifestProject.clips[0],
              updated_at: '2026-04-20T00:02:00Z',
              subtitle_segments: [{ start_time: '00:00:00,000', end_time: '00:00:03,000', text: 'Reconciled subtitle text' }],
              effective_subtitle_text: 'Reconciled subtitle text',
              subtitle_recipe: {},
              has_manual_subtitle_override: false,
              recovery: {},
            }],
          }),
        }
      }
      if (url.endsWith('/clips/clip-real-1/bounds')) {
        return { ok: true, json: async () => ({}) }
      }
      if (url.endsWith('/clips/clip-real-1/rerender/boundary')) {
        return { ok: true, json: async () => ({ job_id: 'job-boundary-2' }) }
      }
      if (url === '/api/jobs/job-boundary-2') {
        return { ok: true, json: async () => ({ status: 'completed' }) }
      }
      throw new Error(`Unexpected fetch in test: ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<App />)

    await screen.findByRole('heading', { name: 'Loaded Clip' })
    const startSlider = screen.getByLabelText(zh('clipStart'))
    fireEvent.change(startSlider, { target: { value: '6' } })
    await user.click(screen.getByRole('button', { name: zh('queueRerender', { operation: zh('operationBoundary') }) }))

    await waitFor(() => expect(screen.getByDisplayValue('Reconciled subtitle text')).toBeInTheDocument())
    const previewPanel = screen.getByRole('article', { name: zh('postProcessedPreview') })
    const previewVideo = previewPanel.querySelector('video')
    expect(previewVideo).toHaveAttribute('src', '/api/projects/proj-real/media/clips_post_processed/clip-real-1.mp4?v=2026-04-20T00%3A02%3A00Z')
    expect(screen.getByRole('button', { name: zh('queueRerender', { operation: zh('operationBoundary') }) })).toBeDisabled()
    expect(screen.queryByText(zh('boundaryRerenderStarted'))).toBeNull()
  })

  it('shows a clear empty state when no post-processed clip is available', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        ...manifestProject,
        clips: [{
          ...manifestProject.clips[0],
          asset_registry: {},
          current_composed_clip_url: undefined,
        }],
      }),
    }))

    render(<App />)

    await screen.findByRole('heading', { name: 'Loaded Clip' })
    const previewPanel = screen.getByRole('article', { name: zh('postProcessedPreview') })
    expect(previewPanel).toHaveTextContent(zh('postProcessedPreviewUnavailable'))
    expect(previewPanel.querySelector('video')).toBeNull()
  })

  it('shows the actual cover assets when available', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => manifestProject }))

    render(<App />)

    await screen.findByRole('heading', { name: 'Loaded Clip' })
    expect(screen.getByAltText(zh('horizontalCoverAlt', { title: 'Loaded cover' }))).toHaveAttribute('src', '/api/projects/proj-real/media/covers/cover-clip-real-1-horizontal.jpg?v=2026-04-20T00%3A00%3A00Z')
    expect(screen.getByAltText(zh('verticalCoverAlt', { title: 'Loaded cover' }))).toHaveAttribute('src', '/api/projects/proj-real/media/covers/cover-clip-real-1-vertical.jpg?v=2026-04-20T00%3A00%3A00Z')
  })

  it('shows clear empty states when cover assets are unavailable', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        ...manifestProject,
        clips: [{
          ...manifestProject.clips[0],
          horizontal_cover_url: undefined,
          vertical_cover_url: undefined,
        }],
      }),
    }))

    render(<App />)

    await screen.findByRole('heading', { name: 'Loaded Clip' })
    expect(screen.getByText(zh('horizontalCoverUnavailable'))).toBeInTheDocument()
    expect(screen.getByText(zh('verticalCoverUnavailable'))).toBeInTheDocument()
  })
})
