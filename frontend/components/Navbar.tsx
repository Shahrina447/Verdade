'use client'

import Link from 'next/link'
import { useState } from 'react'
import { Brain, Menu, X, User } from 'lucide-react'

// Profile info — update these to match your details
const PROFILE = {
  name: 'Shahrina447',
  role: 'Developer',
  avatar: null as string | null, // set to an image URL if you have one
}

export default function Navbar() {
  const [open, setOpen] = useState(false)
  const [profileOpen, setProfileOpen] = useState(false)

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 glass border-b border-white/5">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2">
            <div className="w-9 h-9 rounded-xl btn-primary flex items-center justify-center shadow-lg">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold gradient-text">SachAI</span>
          </Link>

          {/* Desktop links */}
          <div className="hidden md:flex items-center gap-8 text-sm text-slate-400">
            <Link href="/#detector" className="hover:text-white transition-colors">Detector</Link>
            <Link href="/#how-it-works" className="hover:text-white transition-colors">How It Works</Link>
            <Link href="/#features" className="hover:text-white transition-colors">Features</Link>
            <Link href="/history" className="hover:text-white transition-colors">History</Link>
          </div>

          {/* Right side: Status badge + Profile + mobile toggle */}
          <div className="flex items-center gap-3">
            {/* Live badge */}
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full glass text-xs text-slate-300">
              <span className="pulse-dot" />
              <span>Live</span>
            </div>

            {/* Profile button */}
            <div className="relative">
              <button
                onClick={() => setProfileOpen(!profileOpen)}
                className="flex items-center gap-2 px-2 py-1.5 rounded-xl glass border border-white/10 hover:border-purple-500/40 transition-colors"
                aria-label="Profile"
              >
                {PROFILE.avatar ? (
                  <img
                    src={PROFILE.avatar}
                    alt={PROFILE.name}
                    className="w-7 h-7 rounded-full object-cover"
                  />
                ) : (
                  <div className="w-7 h-7 rounded-full btn-primary flex items-center justify-center">
                    <User className="w-3.5 h-3.5 text-white" />
                  </div>
                )}
                <span className="hidden sm:block text-xs text-slate-300 font-medium">
                  {PROFILE.name}
                </span>
              </button>

              {/* Profile dropdown */}
              {profileOpen && (
                <>
                  {/* Backdrop to close on outside click */}
                  <div
                    className="fixed inset-0 z-10"
                    onClick={() => setProfileOpen(false)}
                  />
                  <div className="absolute right-0 mt-2 w-52 glass rounded-xl border border-white/10 shadow-xl z-20 overflow-hidden">
                    {/* Profile info */}
                    <div className="px-4 py-3 border-b border-white/5">
                      <div className="flex items-center gap-3">
                        <div className="w-9 h-9 rounded-full btn-primary flex items-center justify-center shrink-0">
                          <User className="w-4 h-4 text-white" />
                        </div>
                        <div>
                          <p className="text-sm font-semibold text-white">{PROFILE.name}</p>
                          <p className="text-xs text-slate-500">{PROFILE.role}</p>
                        </div>
                      </div>
                    </div>
                    {/* Dropdown actions */}
                    <div className="py-1">
                      <Link
                        href="/history"
                        onClick={() => setProfileOpen(false)}
                        className="flex items-center gap-2.5 px-4 py-2.5 text-xs text-slate-300 hover:bg-white/5 hover:text-white transition-colors"
                      >
                        My History
                      </Link>
                      <Link
                        href="/#detector"
                        onClick={() => setProfileOpen(false)}
                        className="flex items-center gap-2.5 px-4 py-2.5 text-xs text-slate-300 hover:bg-white/5 hover:text-white transition-colors"
                      >
                        New Analysis
                      </Link>
                    </div>
                  </div>
                </>
              )}
            </div>

            {/* Mobile menu toggle */}
            <button
              className="md:hidden p-2 rounded-lg glass"
              onClick={() => setOpen(!open)}
              aria-label="Toggle menu"
            >
              {open ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Mobile menu */}
        {open && (
          <div className="md:hidden py-4 border-t border-white/5 flex flex-col gap-3 text-sm text-slate-400">
            <Link href="/#detector" onClick={() => setOpen(false)} className="hover:text-white transition-colors px-2">Detector</Link>
            <Link href="/#how-it-works" onClick={() => setOpen(false)} className="hover:text-white transition-colors px-2">How It Works</Link>
            <Link href="/#features" onClick={() => setOpen(false)} className="hover:text-white transition-colors px-2">Features</Link>
            <Link href="/history" onClick={() => setOpen(false)} className="hover:text-white transition-colors px-2">History</Link>
          </div>
        )}
      </div>
    </nav>
  )
}
