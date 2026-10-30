export interface ClipDraft {
  id: string
  order: number
  title: string
  sourcePart: string
  start: number
  end: number
  localStart: number
  localEnd: number
  absoluteTimeRange?: string
  localTimeRange?: string
  partAbsoluteStart?: number
  partAbsoluteEnd?: number
  speed: number
  subtitleText: string
  subtitleSegments: SubtitleSegmentDraft[]
  translatedSubtitleText: string
  translatedSubtitleSegments: SubtitleSegmentDraft[]
  hasTranslatedSubtitles: boolean
  coverTitle: string
  renderStatus: 'Ready' | 'Needs sync' | 'Rendering' | 'Recoverable' | 'Error'
  updatedAt: string
  sourceVideoUrl?: string
  currentComposedClip?: string
  currentComposedClipUrl?: string
  rawClip?: string
  rawClipUrl?: string
  horizontalCover?: string
  horizontalCoverUrl?: string
  verticalCover?: string
  verticalCoverUrl?: string
  coverDirty?: boolean
  recoveryState?: string
  lastError?: string
  pendingJobId?: string
  pendingOperation?: string
}

export interface EditorProject {
  projectId: string
  projectName: string
  sourceLabel: string
  sourceVideoUrl?: string
  totalDuration: number
  clips: ClipDraft[]
}

export interface DirtyState {
  hasChanges: boolean
  boundsDirty: boolean
  speedDirty: boolean
  boundaryDirty: boolean
  sourceSubtitlesDirty: boolean
  translatedSubtitlesDirty: boolean
  subtitlesDirty: boolean
  coverTitleDirty: boolean
  coverNeedsRefresh: boolean
}

export interface SubtitleSegmentDraft {
  index: number
  startTime: string
  endTime: string
  text: string
}

interface ManifestAssetRegistry {
  current_composed_clip?: string
  raw_clip?: string
  horizontal_cover?: string
  vertical_cover?: string
}

interface ManifestRecovery {
  pending_job_id?: string
  pending_operation?: string
  dirty?: boolean
  last_error?: string
  last_reconciled_at?: string
  recovery_state?: string
}

interface ManifestClipRecipe {
  override_text?: string
  text?: string
}

interface ManifestSubtitleSegment {
  start_time?: string
  end_time?: string
  text?: string
}

interface ManifestClipMetadata {
  cover_dirty?: boolean
}

interface ManifestClip {
  clip_id: string
  title: string
  video_part?: string
  time_range?: string
  start_time: string
  end_time: string
  absolute_start_time?: string
  absolute_end_time?: string
  absolute_time_range?: string
  speed?: number
  subtitle_recipe?: ManifestClipRecipe
  subtitle_segments?: ManifestSubtitleSegment[]
  effective_subtitle_text?: string
  translated_subtitle_segments?: ManifestSubtitleSegment[]
  translated_subtitle_text?: string
  has_translated_subtitles?: boolean
  has_manual_subtitle_override?: boolean
  cover_recipe?: ManifestClipRecipe
  recovery?: ManifestRecovery
  asset_registry?: ManifestAssetRegistry
  metadata?: ManifestClipMetadata
  source_video_url?: string
  part_offset_seconds?: number
  part_duration_seconds?: number
  current_composed_clip_url?: string
  raw_clip_url?: string
  horizontal_cover_url?: string
  vertical_cover_url?: string
  updated_at?: string
}

interface EditorManifest {
  project_id: string
  source_video_title?: string
  source_video_path?: string
  source_video_url?: string
  project_root?: string
  source_video_duration?: number | string
  updated_at?: string
  clips?: ManifestClip[]
}

const minimumClipLength = 1

export function getProjectIdFromPath(pathname: string): string {
  const segments = pathname.split('/').filter(Boolean)
  const projectIndex = segments.findIndex((segment) => segment === 'projects')
  if (projectIndex >= 0 && segments[projectIndex + 1]) {
    return decodeURIComponent(segments[projectIndex + 1])
  }
  return ''
}

export function parseTimeToSeconds(time: string): number {
  const cleaned = time.replace(',', '.')
  const parts = cleaned.split(':').map((part) => Number(part))
  if (parts.some(Number.isNaN)) return 0
  let seconds = 0
  for (const part of parts) {
    seconds = seconds * 60 + part
  }
  return Number(seconds.toFixed(3))
}


export function formatSecondsForApi(totalSeconds: number): string {
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const seconds = Math.floor(totalSeconds % 60)
  const milliseconds = Math.round((totalSeconds % 1) * 1000)
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(milliseconds).padStart(3, '0')}`
}

export function formatTimestamp(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = Math.floor(totalSeconds % 60)
  const tenths = Math.round((totalSeconds % 1) * 10)
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${tenths}`
}

export function formatApiTimestamp(totalSeconds: number): string {
  const totalMs = Math.round(totalSeconds * 1000)
  const hours = Math.floor(totalMs / 3_600_000)
  const minutes = Math.floor((totalMs % 3_600_000) / 60_000)
  const seconds = Math.floor((totalMs % 60_000) / 1000)
  const milliseconds = totalMs % 1000
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(milliseconds).padStart(3, '0')}`
}

export function clampBounds(start: number, end: number, duration: number): { start: number; end: number } {
  const normalizedStart = Math.max(0, Math.min(start, duration - minimumClipLength))
  const normalizedEnd = Math.max(normalizedStart + minimumClipLength, Math.min(end, duration))
  return { start: Number(normalizedStart.toFixed(3)), end: Number(normalizedEnd.toFixed(3)) }
}

export function clampBoundsWithinRange(
  start: number,
  end: number,
  duration: number,
  minStart: number,
  maxEnd: number,
): { start: number; end: number } {
  const base = clampBounds(start, end, duration)
  const boundedStart = Math.max(base.start, minStart)
  const boundedEnd = Math.min(base.end, maxEnd)
  if (boundedEnd <= boundedStart) {
    const rangeSpan = Math.max(maxEnd - minStart, 0)
    if (rangeSpan <= minimumClipLength) {
      return {
        start: Number(minStart.toFixed(3)),
        end: Number(maxEnd.toFixed(3)),
      }
    }
    const adjustedStart = Math.min(boundedStart, maxEnd - minimumClipLength)
    return {
      start: Number(adjustedStart.toFixed(3)),
      end: Number(Math.min(maxEnd, adjustedStart + minimumClipLength).toFixed(3)),
    }
  }
  return {
    start: Number(boundedStart.toFixed(3)),
    end: Number(boundedEnd.toFixed(3)),
  }
}

export function getDirtyState(savedClip: ClipDraft, draftClip: ClipDraft): DirtyState {
  const boundsDirty = savedClip.start !== draftClip.start || savedClip.end !== draftClip.end
  const speedDirty = savedClip.speed !== draftClip.speed
  const sourceSubtitlesDirty = serializeSubtitleSegments(savedClip.subtitleSegments) !== serializeSubtitleSegments(draftClip.subtitleSegments)
  const translatedSubtitlesDirty = serializeSubtitleSegments(savedClip.translatedSubtitleSegments) !== serializeSubtitleSegments(draftClip.translatedSubtitleSegments)
  const subtitlesDirty = sourceSubtitlesDirty || translatedSubtitlesDirty
  const coverTitleDirty = savedClip.coverTitle !== draftClip.coverTitle
  return {
    hasChanges: boundsDirty || speedDirty || subtitlesDirty || coverTitleDirty,
    boundsDirty,
    speedDirty,
    boundaryDirty: boundsDirty || speedDirty,
    sourceSubtitlesDirty,
    translatedSubtitlesDirty,
    subtitlesDirty,
    coverTitleDirty,
    coverNeedsRefresh: Boolean(draftClip.coverDirty) || boundsDirty || speedDirty || coverTitleDirty,
  }
}

export function subtitleSegmentsToText(segments: SubtitleSegmentDraft[]): string {
  return segments
    .map((segment) => segment.text.trim())
    .filter(Boolean)
    .join('\n')
    .trim()
}

export function serializeSubtitleSegments(segments: SubtitleSegmentDraft[]): string {
  return JSON.stringify(
    segments.map((segment) => ({
      startTime: segment.startTime,
      endTime: segment.endTime,
      text: segment.text,
    })),
  )
}

function mapSubtitleSegments(segments?: ManifestSubtitleSegment[]): SubtitleSegmentDraft[] {
  return (segments ?? []).map((segment, index) => ({
    index: index + 1,
    startTime: segment.start_time ?? '00:00:00,000',
    endTime: segment.end_time ?? '00:00:00,500',
    text: segment.text ?? '',
  }))
}

export function summarizeDirtyState(state: DirtyState): string {
  if (!state.hasChanges && !state.coverNeedsRefresh) {
    return 'No local changes'
  }
  const tokens = []
  if (state.boundsDirty) tokens.push('bounds')
  if (state.speedDirty) tokens.push('speed')
  if (state.subtitlesDirty) tokens.push('subtitles')
  if (state.coverNeedsRefresh) tokens.push('cover')
  return `Unsaved ${tokens.join(' + ')}`
}

function mapRenderStatus(recovery?: ManifestRecovery, coverDirty?: boolean): ClipDraft['renderStatus'] {
  if (recovery?.pending_job_id) return 'Rendering'
  if (recovery?.recovery_state === 'recoverable') return 'Recoverable'
  if (recovery?.recovery_state === 'failed' || recovery?.recovery_state === 'cancelled') return 'Error'
  if (recovery?.dirty || coverDirty) return 'Needs sync'
  return 'Ready'
}

export function emptyProject(projectId: string): EditorProject {
  return {
    projectId,
    projectName: '',
    sourceLabel: '',
    sourceVideoUrl: undefined,
    totalDuration: 0,
    clips: [],
  }
}

export function projectFromManifest(manifest: EditorManifest): EditorProject {
  return {
    projectId: manifest.project_id,
    projectName: manifest.source_video_title ?? manifest.project_id,
    sourceLabel: manifest.source_video_path ?? manifest.project_root ?? '',
    sourceVideoUrl: manifest.source_video_url,
    totalDuration: Number(manifest.source_video_duration ?? 0),
    clips: (manifest.clips ?? []).map((clip, index) => {
      const coverDirty = Boolean(clip.metadata?.cover_dirty)
      const subtitleSegments = mapSubtitleSegments(clip.subtitle_segments)
      const translatedSubtitleSegments = mapSubtitleSegments(clip.translated_subtitle_segments)
      const subtitleText = subtitleSegmentsToText(subtitleSegments) || (clip.effective_subtitle_text ?? clip.subtitle_recipe?.override_text ?? '')
      const translatedSubtitleText = subtitleSegmentsToText(translatedSubtitleSegments) || (clip.translated_subtitle_text ?? '')
      return {
        id: clip.clip_id,
        order: index + 1,
        title: clip.title,
        sourcePart: clip.video_part || '',
        start: parseTimeToSeconds(clip.absolute_start_time ?? clip.start_time),
        end: parseTimeToSeconds(clip.absolute_end_time ?? clip.end_time),
        localStart: parseTimeToSeconds(clip.start_time),
        localEnd: parseTimeToSeconds(clip.end_time),
        absoluteTimeRange: clip.absolute_time_range ?? clip.time_range,
        localTimeRange: clip.time_range,
        partAbsoluteStart: Number(clip.part_offset_seconds ?? 0),
        partAbsoluteEnd: clip.part_duration_seconds !== undefined && clip.part_duration_seconds !== null
          ? Number(clip.part_offset_seconds ?? 0) + Number(clip.part_duration_seconds)
          : undefined,
        speed: Number(clip.speed ?? 1),
        subtitleText,
        subtitleSegments,
        translatedSubtitleText,
        translatedSubtitleSegments,
        hasTranslatedSubtitles: Boolean(clip.has_translated_subtitles),
        coverTitle: clip.cover_recipe?.text ?? clip.title,
        renderStatus: mapRenderStatus(clip.recovery, coverDirty),
        updatedAt: clip.updated_at ?? clip.recovery?.last_reconciled_at ?? manifest.updated_at ?? 'pending manifest sync',
        sourceVideoUrl: clip.source_video_url ?? manifest.source_video_url,
        currentComposedClip: clip.asset_registry?.current_composed_clip,
        currentComposedClipUrl: clip.current_composed_clip_url,
        rawClip: clip.asset_registry?.raw_clip,
        rawClipUrl: clip.raw_clip_url,
        horizontalCover: clip.asset_registry?.horizontal_cover,
        horizontalCoverUrl: clip.horizontal_cover_url,
        verticalCover: clip.asset_registry?.vertical_cover,
        verticalCoverUrl: clip.vertical_cover_url,
        coverDirty,
        recoveryState: clip.recovery?.recovery_state,
        lastError: clip.recovery?.last_error,
        pendingJobId: clip.recovery?.pending_job_id,
        pendingOperation: clip.recovery?.pending_operation,
      }
    }),
  }
}
