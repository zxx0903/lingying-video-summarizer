import { describe, expect, it } from 'vitest'
import { clampBoundsWithinRange, formatSecondsForApi, projectFromManifest } from './editorState'

describe('editorState helpers', () => {
  it('preserves fractional timeline precision for API payloads', () => {
    expect(formatSecondsForApi(12.5)).toBe('00:00:12.500')
    expect(formatSecondsForApi(62.125)).toBe('00:01:02.125')
  })

  it('uses absolute full-video timing as the primary editor timebase while retaining part-local timing as secondary info', () => {
    const project = projectFromManifest({
      project_id: 'proj-1',
      source_video_title: 'Demo',
      source_video_duration: 600,
      clips: [
        {
          clip_id: 'clip-1',
          title: 'Moment',
          video_part: 'part03',
          start_time: '00:00:10',
          end_time: '00:00:25',
          absolute_start_time: '00:08:10',
          absolute_end_time: '00:08:25',
          absolute_time_range: '00:08:10 - 00:08:25',
          time_range: '00:00:10 - 00:00:25',
          asset_registry: {},
          metadata: {},
        },
      ],
    })

    expect(project.clips[0].start).toBe(490)
    expect(project.clips[0].end).toBe(505)
    expect(project.clips[0].localStart).toBe(10)
    expect(project.clips[0].localEnd).toBe(25)
    expect(project.clips[0].localTimeRange).toBe('00:00:10 - 00:00:25')
  })

  it('clamps bounds within the source-part absolute range for draggable timeline edits', () => {
    expect(clampBoundsWithinRange(50, 90, 600, 60, 85)).toEqual({ start: 60, end: 85 })
    expect(clampBoundsWithinRange(70, 120, 600, 60, 85)).toEqual({ start: 70, end: 85 })
    expect(clampBoundsWithinRange(85, 85, 600, 60, 85)).toEqual({ start: 84, end: 85 })
  })
})
