//////////////////////////////////////////////////////////////////////////
//  Plog - portable and simple log for C++
//  Documentation and sources: https://github.com/SergiusTheBest/plog
//  License: MPL 2.0, http://mozilla.org/MPL/2.0/

#pragma once
#include <plog/Record.h>
#include <plog/Logger.h>
#include <plog/Init.h>
#include <plog/Compatibility.h>

//////////////////////////////////////////////////////////////////////////
// Helper macros that get context info

#ifdef _MSC_BUILD
#   if _MSC_VER >= 1600 && !defined(__INTELLISENSE__) // >= Visual Studio 2010 and skip IntelliSense
#       define PLOG_GET_THIS()      __if_exists(this) { this } __if_not_exists(this) { 0 } 
#   else
#       define PLOG_GET_THIS()      0
#   endif
#   define PLOG_GET_FUNC()          __FUNCTION__
#else
#   define PLOG_GET_THIS()          0
#   define PLOG_GET_FUNC()          __PRETTY_FUNCTION__
#endif

//////////////////////////////////////////////////////////////////////////
// Log severity level checker

#define IF_LOG_(instance, severity)     if (plog::get<instance>() && plog::get<instance>()->checkSeverity(severity))
#define IF_LOG(severity)                IF_LOG_(0, severity)

//////////////////////////////////////////////////////////////////////////
// Main logging macros

#define LOOG_(instance, severity)        IF_LOG_(instance, severity) (*plog::get<instance>()) += plog::Record(severity, PLOG_GET_FUNC(), __LINE__, PLOG_GET_THIS())
#define LOOG(severity)                   LOOG_(0, severity)

#define LOOG_VERBOSE                     LOOG(plog::verbose)
#define LOOG_DEBUG                       LOOG(plog::debug)
#define LOOG_INFO                        LOOG(plog::info)
#define LOOG_WARNING                     LOOG(plog::warning)
#define LOOG_ERROR                       LOOG(plog::error)
#define LOOG_FATAL                       LOOG(plog::fatal)

#define LOOG_VERBOSE_(instance)          LOOG_(instance, plog::verbose)
#define LOOG_DEBUG_(instance)            LOOG_(instance, plog::debug)
#define LOOG_INFO_(instance)             LOOG_(instance, plog::info)
#define LOOG_WARNING_(instance)          LOOG_(instance, plog::warning)
#define LOOG_ERROR_(instance)            LOOG_(instance, plog::error)
#define LOOG_FATAL_(instance)            LOOG_(instance, plog::fatal)

#define LOOGV                            LOOG_VERBOSE
#define LOOGD                            LOOG_DEBUG
#define LOOGI                            LOOG_INFO
#define LOOGW                            LOOG_WARNING
#define LOOGE                            LOOG_ERROR
#define LOOGF                            LOOG_FATAL

#define LOOGV_(instance)                 LOOG_VERBOSE_(instance)
#define LOOGD_(instance)                 LOOG_DEBUG_(instance)
#define LOOGI_(instance)                 LOOG_INFO_(instance)
#define LOOGW_(instance)                 LOOG_WARNING_(instance)
#define LOOGE_(instance)                 LOOG_ERROR_(instance)
#define LOOGF_(instance)                 LOOG_FATAL_(instance)

//////////////////////////////////////////////////////////////////////////
// Conditional logging macros

#define LOOG_IF_(instance, severity, condition)  if (condition) LOOG_(instance, severity)
#define LOOG_IF(severity, condition)             LOOG_IF_(0, severity, condition)

#define LOOG_VERBOSE_IF(condition)               LOOG_IF(plog::verbose, condition)
#define LOOG_DEBUG_IF(condition)                 LOOG_IF(plog::debug, condition)
#define LOOG_INFO_IF(condition)                  LOOG_IF(plog::info, condition)
#define LOOG_WARNING_IF(condition)               LOOG_IF(plog::warning, condition)
#define LOOG_ERROR_IF(condition)                 LOOG_IF(plog::error, condition)
#define LOOG_FATAL_IF(condition)                 LOOG_IF(plog::fatal, condition)

#define LOOG_VERBOSE_IF_(instance, condition)    LOOG_IF_(instance, plog::verbose, condition)
#define LOOG_DEBUG_IF_(instance, condition)      LOOG_IF_(instance, plog::debug, condition)
#define LOOG_INFO_IF_(instance, condition)       LOOG_IF_(instance, plog::info, condition)
#define LOOG_WARNING_IF_(instance, condition)    LOOG_IF_(instance, plog::warning, condition)
#define LOOG_ERROR_IF_(instance, condition)      LOOG_IF_(instance, plog::error, condition)
#define LOOG_FATAL_IF_(instance, condition)      LOOG_IF_(instance, plog::fatal, condition)

#define LOOGV_IF(condition)                      LOOG_VERBOSE_IF(condition)
#define LOOGD_IF(condition)                      LOOG_DEBUG_IF(condition)
#define LOOGI_IF(condition)                      LOOG_INFO_IF(condition)
#define LOOGW_IF(condition)                      LOOG_WARNING_IF(condition)
#define LOOGE_IF(condition)                      LOOG_ERROR_IF(condition)
#define LOOGF_IF(condition)                      LOOG_FATAL_IF(condition)

#define LOOGV_IF_(instance, condition)           LOOG_VERBOSE_IF_(instance, condition)
#define LOOGD_IF_(instance, condition)           LOOG_DEBUG_IF_(instance, condition)
#define LOOGI_IF_(instance, condition)           LOOG_INFO_IF_(instance, condition)
#define LOOGW_IF_(instance, condition)           LOOG_WARNING_IF_(instance, condition)
#define LOOGE_IF_(instance, condition)           LOOG_ERROR_IF_(instance, condition)
#define LOOGF_IF_(instance, condition)           LOOG_FATAL_IF_(instance, condition)

