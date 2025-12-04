"""
Appointment Manager Agent.

Handles appointment booking, rescheduling, cancellation, and checking.
"""

import logging
import asyncio
import hashlib
import time as time_module
import concurrent.futures
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from datetime import datetime as dt

from .base_agent import BaseAgent
from patient_ai_service.infrastructure.db_ops_client import DbOpsClient

logger = logging.getLogger(__name__)


class AppointmentManagerAgent(BaseAgent):
    """
    Agent responsible for all appointment-related operations.

    Features:
    - Book new appointments
    - Reschedule existing appointments
    - Cancel appointments
    - Check appointment status
    - Find available time slots
    """

    def __init__(self, db_client: Optional[DbOpsClient] = None, **kwargs):
        # Extract max_tool_iterations from kwargs if present (not used by BaseAgent)
        max_tool_iterations = kwargs.pop('max_tool_iterations', 15)  # Higher for complex booking flows
        super().__init__(agent_name="AppointmentManager", **kwargs)
        self.db_client = db_client or DbOpsClient()
        self.max_tool_iterations = max_tool_iterations

    async def on_activated(self, session_id: str, reasoning: Any):
        """
        Set up appointment workflow when agent is activated.

        Args:
            session_id: Session identifier
            reasoning: ReasoningOutput from reasoning engine
        """
        import json
        
        # Log the reasoning output received by appointment manager
        logger.info("=" * 80)
        logger.info("APPOINTMENT_MANAGER: on_activated() - Reasoning Output Received")
        logger.info("=" * 80)
        logger.info(f"Session: {session_id}")
        logger.info(f"Routing:")
        logger.info(f"  - agent: {reasoning.routing.agent}")
        logger.info(f"  - action: {reasoning.routing.action}")
        logger.info(f"  - urgency: {reasoning.routing.urgency}")
        logger.info(f"Understanding:")
        logger.info(f"  - what_user_means: {reasoning.understanding.what_user_means}")
        logger.info(f"  - sentiment: {reasoning.understanding.sentiment}")
        logger.info(f"  - is_continuation: {reasoning.understanding.is_continuation}")
        logger.info(f"Memory Updates:")
        logger.info(f"  - system_action: {reasoning.memory_updates.system_action or '(empty)'}")
        logger.info(f"  - awaiting: {reasoning.memory_updates.awaiting or '(empty)'}")
        if reasoning.memory_updates.new_facts:
            logger.info(f"  - new_facts: {json.dumps(reasoning.memory_updates.new_facts, indent=2)}")
        if reasoning.response_guidance.minimal_context:
            logger.info(f"Response Guidance: {json.dumps(reasoning.response_guidance.minimal_context, indent=2)}")
        if reasoning.response_guidance.plan:
            logger.info("=" * 80)
            logger.info("📋 PLAN FOR APPOINTMENT MANAGER:")
            logger.info("=" * 80)
            logger.info(reasoning.response_guidance.plan)
            logger.info("=" * 80)
        logger.info("=" * 80)
        
        # Determine operation type from routing action
        action = reasoning.routing.action.lower() if reasoning.routing.action else ""

        operation_type = "booking"  # Default
        if "reschedule" in action or "change" in action:
            operation_type = "rescheduling"
        elif "cancel" in action:
            operation_type = "cancellation"
        elif "check" in action or "view" in action or "list" in action:
            operation_type = "checking"
        elif "book" in action or "schedule" in action or "appointment" in action:
            operation_type = "booking"

        # Store the plan from reasoning for use in system prompt
        plan_from_reasoning = reasoning.response_guidance.plan if reasoning.response_guidance.plan else ""
        
        # Initialize appointment workflow state
        self.state_manager.update_appointment_state(
            session_id,
            workflow_step="gathering_info",
            operation_type=operation_type,
            reasoning_plan=plan_from_reasoning  # Store plan in state
        )

        logger.info(f"Appointment workflow initialized: operation_type={operation_type}, session={session_id}")

    def _register_tools(self):
        """Register appointment-related tools."""

        # Get doctors list
        self.register_tool(
            name="list_doctors",
            function=self.tool_list_doctors,
            description="Get list of available doctors with their specialties and languages",
            parameters={}
        )

        # Check doctor availability
        # Calculate dates dynamically for the description
        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)
        today_str = today.strftime('%Y-%m-%d')
        tomorrow_str = tomorrow.strftime('%Y-%m-%d')
        
        self.register_tool(
            name="check_availability",
            function=self.tool_check_availability,
            description=f"Check doctor availability with TWO MODES:\n\n1. RANGE MODE (no requested_time): Returns availability_ranges - continuous time blocks where the doctor is available. Example: ['9:00-10:00', '10:30-15:00', '15:30-17:00']. Use these ranges to suggest appointment times naturally.\n\n2. SPECIFIC TIME MODE (with requested_time): Checks if a specific time is available. Returns true/false and alternatives if unavailable. Example: requested_time='14:00' returns available_at_requested_time=true/false plus alternative_slots=['14:30', '15:00', '13:30'] if false.\n\nIMPORTANT: doctor_id must be a UUID (use list_doctors first to get it), not a doctor name. When parsing 'tomorrow', calculate it dynamically: today is {today_str}, so tomorrow is {tomorrow_str}.",
            parameters={
                "doctor_id": {
                    "type": "string",
                    "description": "Doctor's UUID (NOT name - use list_doctors tool first to get the UUID)"
                },
                "date": {
                    "type": "string",
                    "description": f"Date in YYYY-MM-DD format (e.g., '{tomorrow_str}'). Parse 'tomorrow' to actual date based on today ({today_str}). Always calculate tomorrow dynamically from the current date."
                },
                "requested_time": {
                    "type": "string",
                    "description": "Optional: Specific time to check (e.g., '14:00', '2pm', '2:00 PM'). If provided, returns whether that specific time is available plus alternatives if not. If omitted, returns availability ranges."
                }
            }
        )

        # Book appointment
        self.register_tool(
            name="book_appointment",
            function=self.tool_book_appointment,
            description="Book a new appointment. MANDATORY: When check_availability returns 'MANDATORY_ACTION': 'CALL book_appointment TOOL IMMEDIATELY', you MUST call this tool immediately. Do NOT generate text - make the tool call. Extract the reason from user's message (their exact wording). Only use 'general consultation' if user did not mention any reason.",
            parameters={
                "patient_id": {
                    "type": "string",
                    "description": "Patient's ID (get from Patient ID in PATIENT INFORMATION section - REQUIRED)"
                },
                "doctor_id": {
                    "type": "string",
                    "description": "Doctor's UUID (from find_doctor_by_name tool result - REQUIRED)"
                },
                "date": {
                    "type": "string",
                    "description": "Appointment date in YYYY-MM-DD format (e.g., '2025-11-26' - REQUIRED)"
                },
                "time": {
                    "type": "string",
                    "description": "Appointment time in HH:MM format (e.g., '15:00' for 3:00 PM - REQUIRED)"
                },
                "reason": {
                    "type": "string",
                    "description": "Extract the EXACT reason/procedure/symptom from user's message. Use user's own words. ONLY use 'general consultation' if user did not mention any specific reason."
                }
            }
        )

        # Check patient appointments
        self.register_tool(
            name="check_patient_appointments",
            function=self.tool_check_patient_appointments,
            description="Get list of patient's appointments",
            parameters={
                "patient_id": {
                    "type": "string",
                    "description": "Patient's ID"
                }
            }
        )

        # Cancel appointment
        self.register_tool(
            name="cancel_appointment",
            function=self.tool_cancel_appointment,
            description="Cancel an existing appointment",
            parameters={
                "appointment_id": {
                    "type": "string",
                    "description": "Appointment ID to cancel"
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for cancellation"
                }
            }
        )

        # Reschedule appointment
        self.register_tool(
            name="reschedule_appointment",
            function=self.tool_reschedule_appointment,
            description="Reschedule an existing appointment",
            parameters={
                "appointment_id": {
                    "type": "string",
                    "description": "Appointment ID to reschedule"
                },
                "new_date": {
                    "type": "string",
                    "description": "New date (YYYY-MM-DD)"
                },
                "new_time": {
                    "type": "string",
                    "description": "New time (HH:MM)"
                }
            }
        )

        # Update appointment (flexible - can update any combination of fields)
        self.register_tool(
            name="update_appointment",
            function=self.tool_update_appointment,
            description="Update an appointment with one or more parameters. Can update any combination of: doctor, clinic, patient, date, time, status, reason, notes, emergency level, follow-up settings, or procedure type. Only provide the fields you want to change.",
            parameters={
                "appointment_id": {
                    "type": "string",
                    "description": "Appointment ID to update (REQUIRED)"
                },
                "doctor_id": {
                    "type": "string",
                    "description": "Optional: New doctor ID (UUID)"
                },
                "clinic_id": {
                    "type": "string",
                    "description": "Optional: New clinic ID (UUID)"
                },
                "patient_id": {
                    "type": "string",
                    "description": "Optional: New patient ID (UUID)"
                },
                "appointment_type_id": {
                    "type": "string",
                    "description": "Optional: New appointment type ID (UUID)"
                },
                "appointment_date": {
                    "type": "string",
                    "description": "Optional: New appointment date (YYYY-MM-DD format)"
                },
                "start_time": {
                    "type": "string",
                    "description": "Optional: New start time (HH:MM format, 24-hour)"
                },
                "end_time": {
                    "type": "string",
                    "description": "Optional: New end time (HH:MM format, 24-hour). If start_time is provided, end_time will be calculated automatically if not specified."
                },
                "status": {
                    "type": "string",
                    "description": "Optional: New status. Valid values: scheduled, confirmed, checked_in, in_progress, completed, cancelled, no_show, rescheduled"
                },
                "reason": {
                    "type": "string",
                    "description": "Optional: Update appointment reason/description"
                },
                "notes": {
                    "type": "string",
                    "description": "Optional: Update appointment notes"
                },
                "emergency_level": {
                    "type": "string",
                    "description": "Optional: Change emergency level. Valid values: routine, urgent, emergency, critical"
                },
                "follow_up_required": {
                    "type": "boolean",
                    "description": "Optional: Set whether follow-up is required"
                },
                "follow_up_days": {
                    "type": "integer",
                    "description": "Optional: Number of days until follow-up"
                },
                "procedure_type": {
                    "type": "string",
                    "description": "Optional: Update procedure type"
                }
            }
        )

        # Book multiple appointments at once
        self.register_tool(
            name="book_multiple_appointments",
            function=self.tool_book_multiple_appointments,
            description="Book multiple appointments at once with different parameters for each. Use this when the user requests multiple appointments in a single interaction (e.g., 'book 3 appointments', 'book root canal at 3pm and cleaning at 3:30pm'). Each appointment can have different doctor, date, time, and reason.",
            parameters={
                "patient_id": {
                    "type": "string",
                    "description": "Patient's ID (same for all appointments - get from Patient ID in PATIENT INFORMATION section - REQUIRED)"
                },
                "appointments": {
                    "type": "array",
                    "description": "List of appointments to book. Each appointment is an object with doctor_id, date, time, and reason.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "doctor_id": {
                                "type": "string",
                                "description": "Doctor's UUID (from list_doctors tool - REQUIRED)"
                            },
                            "date": {
                                "type": "string",
                                "description": "Appointment date in YYYY-MM-DD format (e.g., '2025-11-26' - REQUIRED)"
                            },
                            "time": {
                                "type": "string",
                                "description": "Appointment time in HH:MM format (e.g., '15:00' for 3:00 PM - REQUIRED)"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Extract the EXACT reason/procedure/symptom from user's message. Use user's own words. ONLY use 'general consultation' if user did not mention any specific reason."
                            }
                        },
                        "required": ["doctor_id", "date", "time", "reason"]
                    }
                }
            }
        )

    def _get_system_prompt(self, session_id: str) -> str:
        """Generate system prompt with context."""
        # Get state
        global_state = self.state_manager.get_global_state(session_id)
        agent_state = self.state_manager.get_appointment_state(session_id)
        patient = global_state.patient_profile

        # Check if patient is registered
        patient_registered = (
            patient.patient_id is not None and
            patient.patient_id != ""
        )
        
        # Calculate actual dates for dynamic examples
        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)
        today_str = today.strftime('%Y-%m-%d')
        tomorrow_str = tomorrow.strftime('%Y-%m-%d')
        current_year = today.year
        current_month = today.month
        
        # Debug logging
        logger.info(f"🔍 AppointmentManager._get_system_prompt - session: {session_id}, patient_id: {patient.patient_id}, registered: {patient_registered}")
        
        registration_status = "✅ Registered" if patient_registered else "❌ Not Registered - Registration Required"
        
        # Get plan from state (stored in on_activated)
        reasoning_plan = getattr(agent_state, 'reasoning_plan', '') or ''
        
        context = f"""You are an appointment manager for Bright Smile Dental Clinic.

PATIENT CONTEXT:
- Name: {patient.first_name or 'Not provided'} {patient.last_name or ''}
- Patient ID: {patient.patient_id or 'None - Registration Required'}
- Registered: {'Yes' if patient.patient_id else 'No - Must register before booking'}
- Phone: {patient.phone or 'Not provided'}
- Language: {patient.preferred_language or 'en'}

{"=" * 80}
REASONING ENGINE GUIDANCE:
{"=" * 80}
{reasoning_plan if reasoning_plan else "No specific plan provided."}
{"=" * 80}

YOUR ORCHESTRATION PHILOSOPHY:
You are a THINKING agent, not a script-follower. You decide:
✓ What tools to call and in what order
✓ How many appointments to book (could be 1, 2, or more)
✓ When all user requirements are satisfied
✓ When to respond to the user

CRITICAL RULES:

1. MULTI-STEP THINKING:
   Before calling any tools, think through the complete plan:
   - What is the user asking for?
   - How many appointments do they need?
   - What sequence of tools will satisfy their request?
   - What validation is needed?

2. TOOL EXECUTION SEQUENCE:
   Example for "book mohammed atef at 3pm for root canal and teeth cleaning afterward":
   
   Step 1: Understand requirements
   - Need 2 appointments: root canal (3pm) and teeth cleaning (3:30pm)
   
   Step 2: Execute first appointment
   - list_doctors → find "Mohammed Atef" UUID
   - check_availability for 3pm
   - book_appointment (root canal, 3pm)
   
   Step 3: Execute second appointment  
   - check_availability for 3:30pm (same doctor)
   - book_appointment (teeth cleaning, 3:30pm)
   
   Step 4: Validate completeness
   - Root canal booked? ✓
   - Teeth cleaning booked? ✓
   - All requirements satisfied? ✓
   
   Step 5: Respond to user
   - "✅ Both appointments confirmed..."

3. COMPLETENESS CHECK:
   Before responding, ALWAYS ask yourself:
   "Did I fulfill EVERYTHING the user requested?"
   
   If NO → Continue executing tools
   If YES → Generate final response
   
   Examples:
   - User: "book 3 appointments" → You must book 3, not 1
   - User: "book and send confirmation email" → Book + email
   - User: "cancel appointment and reschedule" → Cancel + book new

4. RESPONSE GENERATION RULES:
   ✗ DON'T respond until ALL tasks are complete
   ✗ DON'T say "I'll check..." or "Let me..." - just execute silently
   ✗ DON'T generate intermediate status updates
   ✓ DO call all necessary tools first
   ✓ DO validate results
   ✓ DO respond only when everything is done

5. TOOL CALLING BEST PRACTICES:
   - list_doctors first to get UUIDs (use doctor names from user message)
   - check_availability before booking (required)
   - When "MANDATORY_ACTION" appears → call book_appointment immediately
   - Extract reason from user's exact wording (e.g., "root canal", "cleaning")
   - Calculate sequential appointment times (30 min intervals)

6. HANDLING MULTIPLE APPOINTMENTS:
   When user requests multiple procedures, you have TWO options:

   Option A: Use book_multiple_appointments tool (RECOMMENDED)
   - Single tool call for all appointments
   - Automatically handles sequential booking with retries and verification
   - Returns comprehensive results for all bookings
   - Best for: 2+ appointments, especially with different parameters

   Option B: Sequential book_appointment calls
   - Manual flow with individual calls
   - More control over each step
   - Best for: Complex scenarios requiring availability checks between bookings

   Example flow using book_multiple_appointments:
   ```
   [Call list_doctors] → Get doctor UUID
   [Call check_availability for 3pm] → Verify slot available
   [Call check_availability for 3:30pm] → Verify slot available
   [Call book_multiple_appointments with array of 2 appointments] → Books both ✓
   [All done? Yes] → Generate response
   ```

   Example flow using sequential book_appointment:
   ```
   [Call list_doctors] → Get doctor UUID
   [Call check_availability for 3pm] → Available ✓
   [Call book_appointment for root canal at 3pm] → Booked ✓
   [Call check_availability for 3:30pm] → Available ✓
   [Call book_appointment for cleaning at 3:30pm] → Booked ✓
   [All done? Yes] → Generate response
   ```

7. DATE/TIME PARSING:
   - "tomorrow" → {tomorrow_str}
   - "today" → {today_str}
   - "3pm" → "15:00"
   - "11:30 am" → "11:30"
   - "afternoon" → suggest 2pm-5pm slots
   - "afterward" / "consecutively" → Add 30 minutes to previous appointment

8. SEQUENTIAL BOOKING LOGIC:
   If user says "X procedure and Y procedure afterward":
   - First appointment: requested time
   - Second appointment: first_time + 30 minutes
   - Third appointment: second_time + 30 minutes

   Example:
   "root canal at 3pm and cleaning afterward"
   → Root canal: 15:00
   → Cleaning: 15:30

9. ERROR HANDLING:
   If any tool fails:
   - Check if you can still satisfy user request partially
   - If partial fulfillment is acceptable, continue with other tasks
   - If not, inform user about the specific failure
   - Offer alternatives when possible

   With book_multiple_appointments:
   - Tool returns detailed results for each appointment
   - Successful bookings are still confirmed even if some fail
   - Present clear summary of successes and failures to user

AVAILABLE TOOLS:
1. list_doctors - Get all doctors with UUIDs and specialties
2. find_doctor_by_name - Find specific doctor by name (returns UUID)
3. check_availability - Returns availability_ranges (e.g., ["9:00-12:00", "14:00-17:00"]). Present these ranges naturally to users as "9:00 AM - 12:00 PM" and "2:00 PM - 5:00 PM"
4. book_appointment - Create single appointment (requires patient_id, doctor_id, date, time, reason)
5. book_multiple_appointments - Book multiple appointments at once with different parameters
6. check_patient_appointments - View patient's existing appointments
7. cancel_appointment - Cancel an appointment
8. reschedule_appointment - Reschedule to new date/time
9. update_appointment - Modify any appointment field

RESPONSE STYLE:
- Warm, concise, professional
- No verbose explanations
- Clear confirmation of what was done
- Format dates nicely (e.g., "December 3, 2025")
- Format times nicely (e.g., "3:00 PM")
- Use checkmarks (✓) for completed tasks
- Maximum 5-6 sentences in final response

EXAMPLES OF CORRECT BEHAVIOR:

Example 1 - Single Appointment:
User: "mohammed atef tomorrow 11:30 am please"
Agent: [Silent: list_doctors → check_availability → book_appointment]
       "✅ Your appointment with Dr. Mohammed Atef on December 3, 2025 at 11:30 AM is confirmed. Appointment ID: ABC123"

Example 2 - Multiple Appointments (Using book_multiple_appointments):
User: "book mohammed atef at 3pm for root canal and teeth cleaning afterward"
Agent: [Silent: list_doctors → check_availability(3pm) → check_availability(3:30pm)
        → book_multiple_appointments([{"root canal, 3pm"}, {"cleaning, 3:30pm"}])]
       "✅ All 2 appointments confirmed:
        1) Root canal - December 3, 2025 at 3:00 PM (ID: ABC123)
        2) Teeth cleaning - December 3, 2025 at 3:30 PM (ID: DEF456)"

Example 3 - Multiple Appointments (Sequential booking):
User: "book dr. sarah at 2pm for checkup and dr. mohammed at 4pm for cleaning"
Agent: [Silent: list_doctors → check_availability(2pm for sarah) → book(checkup, 2pm, sarah)
        → check_availability(4pm for mohammed) → book(cleaning, 4pm, mohammed)]
       "✅ Both appointments confirmed:
        1) Checkup with Dr. Sarah - December 3, 2025 at 2:00 PM (ID: ABC123)
        2) Cleaning with Dr. Mohammed - December 3, 2025 at 4:00 PM (ID: DEF456)"

Example 4 - Exploration (No Booking):
User: "what doctors do you have?"
Agent: [Calls list_doctors]
       "We have Dr. Mohammed Atef (General Dentist), Dr. Sarah Johnson (Orthodontist)..."

REMEMBER: You are INTELLIGENT and AUTONOMOUS. Think through the complete plan, execute all necessary tools, validate completeness, then respond. Don't rush to respond before all tasks are done.

⚠️ CRITICAL CONFIRMATION RULES ⚠️

1. NEVER say "confirmed", "booked", "scheduled", or "done" until you receive
   a tool result with "success": true AND "verified": true

2. If a tool returns an error, NEVER claim the action succeeded

3. If you're unsure whether an action completed, say:
   "I'm checking on that..." and call the appropriate verification tool

4. For multi-step tasks, confirm EACH step only after its tool succeeds

5. FORBIDDEN PHRASES during tool execution:
   - "Your appointment is confirmed"
   - "I've booked..."
   - "All set!"
   - "Done!"

   ALLOWED PHRASES during tool execution:
   - "Let me check..."
   - "Checking availability..."
   - "Processing your request..."
"""

        return context

    # Tool implementations

    def tool_list_doctors(self, session_id: str) -> Dict[str, Any]:
        """Get list of available doctors."""
        import time as time_module
        start_time = time_module.time()
        
        logger.info("=" * 80)
        logger.info("APPOINTMENT_MANAGER: tool_list_doctors() CALLED")
        logger.info("=" * 80)
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Input: session_id={session_id}")
        
        try:
            doctors = self.db_client.get_doctors()

            if not doctors:
                duration_ms = (time_module.time() - start_time) * 1000
                logger.info(f"Output: {{'error': 'No doctors found'}}")
                logger.info(f"Time taken: {duration_ms:.2f}ms")
                logger.info("=" * 80)
                return {"error": "No doctors found"}

            # Format doctor list
            doctor_list = []
            for doc in doctors:
                doctor_list.append({
                    "id": doc.get("id"),
                    "name": f"Dr. {doc.get('first_name')} {doc.get('last_name')}",
                    "specialty": doc.get("specialty"),
                    "languages": doc.get("languages", ["en"])
                })

            duration_ms = (time_module.time() - start_time) * 1000
            result = {
                "success": True,
                "doctors": doctor_list,
                "count": len(doctor_list)
            }
            
            logger.info(f"Output: success=True, count={len(doctor_list)}")
            logger.info(f"Time taken: {duration_ms:.2f}ms")
            logger.info("=" * 80)
            
            return result

        except Exception as e:
            duration_ms = (time_module.time() - start_time) * 1000
            logger.error(f"Error listing doctors: {e}")
            logger.info(f"Output: {{'error': '{str(e)}'}}")
            logger.info(f"Time taken: {duration_ms:.2f}ms")
            logger.info("=" * 80)
            return {"error": str(e)}

    def tool_find_doctor_by_name(self, session_id: str, doctor_name: str) -> Dict[str, Any]:
        """Find a doctor by name."""
        try:
            # Get all doctors
            doctors = self.db_client.get_doctors()

            if not doctors:
                return {"error": "No doctors found"}

            # Clean doctor name (remove "Dr.", "Dr", etc.)
            doctor_name_clean = doctor_name.lower().replace("dr.", "").replace("dr", "").strip()
            # Split into words for better matching
            search_words = [w.strip() for w in doctor_name_clean.split() if len(w.strip()) > 2]
            
            matching_doctors = []
            
            for doctor in doctors:
                first_name = doctor.get("first_name", "").lower()
                last_name = doctor.get("last_name", "").lower()
                full_name = f"{first_name} {last_name}".strip()
                
                # Multi-word search: all words should match (e.g., "mohammed atef" requires both words)
                if len(search_words) > 1:
                    # Check if all search words are in the full name
                    if all(word in full_name for word in search_words):
                        matching_doctors.append(doctor)
                else:
                    # Single word search: check if it's in first name, last name, or full name
                    search_term = search_words[0] if search_words else doctor_name_clean
                    if (search_term in first_name or 
                        search_term in last_name or 
                        search_term in full_name):
                        matching_doctors.append(doctor)

            if not matching_doctors:
                return {
                    "success": False,
                    "error": f"No doctor found matching '{doctor_name}'",
                    "suggestion": "Use list_doctors tool to see all available doctors"
                }

            # Return the first matching doctor (best match)
            doctor = matching_doctors[0]
            
            return {
                "success": True,
                "doctor": {
                    "id": doctor.get("id"),
                    "name": f"Dr. {doctor.get('first_name')} {doctor.get('last_name')}",
                    "first_name": doctor.get("first_name"),
                    "last_name": doctor.get("last_name"),
                    "specialty": doctor.get("specialty"),
                    "languages": doctor.get("languages", ["en"])
                }
            }

        except Exception as e:
            logger.error(f"Error finding doctor by name: {e}")
            return {"error": str(e)}

    def _merge_timeslots_to_ranges(self, timeslots: List[Dict[str, Any]]) -> List[str]:
        """
        Merge consecutive available timeslots into time ranges.

        This is the primary method for calculating availability ranges from the
        /appointments/:doctorId/time-slots endpoint, which already filters out booked slots.

        Example:
        Input: [
            {"start_time": "10:00", "end_time": "10:30", "is_available": True},
            {"start_time": "10:30", "end_time": "11:00", "is_available": True},
            {"start_time": "15:00", "end_time": "15:30", "is_available": True}
        ]
        Output: ["10:00-11:00", "15:00-15:30"]

        Args:
            timeslots: List of timeslot dicts with start_time, end_time, is_available

        Returns:
            List of merged time ranges in "HH:MM-HH:MM" format
        """
        if not timeslots:
            return []

        # Filter only available slots
        available = [s for s in timeslots if s.get('is_available', False)]
        if not available:
            return []

        # Sort by start time to ensure correct ordering
        available.sort(key=lambda x: x.get('start_time', ''))

        ranges = []
        current_start = available[0]['start_time']
        current_end = available[0]['end_time']

        for slot in available[1:]:
            slot_start = slot['start_time']
            slot_end = slot['end_time']

            # If this slot starts exactly where the previous one ended, merge them
            if slot_start == current_end:
                current_end = slot_end
            else:
                # Gap detected - save current range and start a new one
                ranges.append(f"{current_start}-{current_end}")
                current_start = slot_start
                current_end = slot_end

        # Don't forget the last range
        ranges.append(f"{current_start}-{current_end}")

        return ranges

    def _normalize_time_to_24hr(self, time_str: str) -> Optional[str]:
        """
        Normalize various time formats to 24-hour HH:MM format.

        Supports:
        - "14:00", "14:30" (already normalized)
        - "2pm", "2PM", "2:00pm", "2:00 PM" (12-hour with AM/PM)
        - "2am", "2AM", "2:00am", "2:00 AM"
        - "14", "02" (hour only, assumes :00)

        Args:
            time_str: Time string in various formats

        Returns:
            Normalized time in "HH:MM" format, or None if parsing fails
        """
        import re

        if not time_str:
            return None

        # Strip whitespace and convert to lowercase for easier parsing
        time_str = time_str.strip().lower()

        # Pattern 1: Already in HH:MM or H:MM format (24-hour)
        match = re.match(r'^(\d{1,2}):(\d{2})$', time_str)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return f"{hour:02d}:{minute:02d}"
            else:
                logger.warning(f"Invalid time values: hour={hour}, minute={minute}")
                return None

        # Pattern 2: 12-hour format with AM/PM (e.g., "2pm", "2:30pm", "2:30 PM")
        match = re.match(r'^(\d{1,2}):?(\d{2})?\s*(am|pm)$', time_str)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2)) if match.group(2) else 0
            period = match.group(3)

            # Validate ranges
            if not (1 <= hour <= 12 and 0 <= minute <= 59):
                logger.warning(f"Invalid 12-hour time: hour={hour}, minute={minute}")
                return None

            # Convert to 24-hour format
            if period == 'am':
                if hour == 12:
                    hour = 0  # 12am is 00:00
            else:  # pm
                if hour != 12:
                    hour += 12  # 1pm-11pm become 13-23, 12pm stays 12

            return f"{hour:02d}:{minute:02d}"

        # Pattern 3: Hour only (e.g., "14", "2") - assume :00
        match = re.match(r'^(\d{1,2})$', time_str)
        if match:
            hour = int(match.group(1))
            if 0 <= hour <= 23:
                return f"{hour:02d}:00"
            else:
                logger.warning(f"Invalid hour: {hour}")
                return None

        # Could not parse
        logger.warning(f"Could not parse time format: '{time_str}'")
        return None

    def _is_time_in_available_slots(
        self,
        requested_time: str,
        available_slots: List[Dict[str, Any]]
    ) -> bool:
        """
        Check if a specific time is available in the list of time slots.

        Args:
            requested_time: Time in HH:MM format (normalized)
            available_slots: List of slot dicts with 'start_time' and 'end_time'

        Returns:
            True if the requested time matches any slot's start_time
        """
        if not available_slots:
            return False

        # Check each slot
        for slot in available_slots:
            start_time = slot.get('start_time', '')

            # Extract HH:MM part (handle "14:00:00" or "14:00" formats)
            if len(start_time) >= 5:
                start_time_short = start_time[:5]  # Get first 5 chars (HH:MM)

                if start_time_short == requested_time:
                    return True

        return False


    def _get_alternative_slots(
        self,
        session_id: str,
        doctor_id: str,
        date: str,
        requested_time: str,
        available_timeslots: List[Dict[str, Any]]
    ) -> str:
        """
        Get alternative time slots closest to the requested time when unavailable.

        Ranks available slots by time proximity (absolute difference in minutes).

        Args:
            session_id: Session identifier
            doctor_id: Doctor ID
            date: Requested date
            requested_time: Requested time (HH:MM format)
            available_timeslots: Pre-fetched available slots

        Returns:
            String with up to 4 alternative slots closest to the requested time
        """
        try:
            # Convert time to minutes since midnight for fast comparison
            def time_to_minutes(time_str: str) -> int:
                """Convert HH:MM to minutes since midnight"""
                try:
                    h, m = time_str[:5].split(':')
                    return int(h) * 60 + int(m)
                except (ValueError, IndexError):
                    return -1  # Invalid time

            requested_minutes = time_to_minutes(requested_time)
            if requested_minutes == -1:
                logger.warning(f"Invalid time format '{requested_time}' for session {session_id}")
                # Fallback: return first 4 slots if time is invalid
                slots = [slot.get('start_time', '')[:5] for slot in available_timeslots[:4] if slot.get('start_time')]
                return ", ".join(slots) if slots else "No alternative slots available"

            # Extract valid slots with their time differences
            valid_slots = []
            for slot in available_timeslots:
                start = slot.get('start_time', '')[:5]
                if start:
                    minutes = time_to_minutes(start)
                    if minutes != -1:
                        valid_slots.append((start, abs(minutes - requested_minutes)))

            if not valid_slots:
                logger.info(f"No valid slots found for doctor {doctor_id} on {date} (session: {session_id})")
                return "No alternative slots available"

            # Sort by proximity (time difference) and take top 4
            closest_slots = sorted(valid_slots, key=lambda x: x[1])[:4]

            # Return just the time strings
            result = ", ".join(slot[0] for slot in closest_slots)
            logger.debug(f"Found {len(closest_slots)} alternative slots for session {session_id}: {result}")
            return result
        except Exception as e:
            logger.warning(f"Failed to get alternative slots for session {session_id}: {e}")
            return "No alternative slots available"

    def _check_specific_time_availability(
        self,
        session_id: str,
        doctor_id: str,
        date: str,
        requested_time: str,
        available_timeslots: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check if a specific requested time is available and provide alternatives if not.

        Args:
            session_id: Session identifier
            doctor_id: Doctor ID
            date: Date string
            requested_time: User-provided time (may need normalization)
            available_timeslots: Pre-fetched available slots

        Returns:
            Dict with availability status and alternatives if unavailable
        """
        # Step 1: Normalize the requested time
        normalized_time = self._normalize_time_to_24hr(requested_time)

        if normalized_time is None:
            logger.warning(f"Failed to normalize time '{requested_time}' for session {session_id}")
            return {
                "requested_time": requested_time,
                "available_at_requested_time": False,
                "error": "Invalid time format",
                "message": f"Could not parse time '{requested_time}'. Please use format like '14:00' or '2pm'"
            }

        # Step 2: Check if the normalized time is in available slots
        is_available = self._is_time_in_available_slots(normalized_time, available_timeslots)

        # Step 3a: If available, return success
        if is_available:
            logger.info(f"✓ Time {normalized_time} is available for doctor {doctor_id} on {date} (session: {session_id})")

            # Update state with the checked time
            self.state_manager.update_appointment_state(
                session_id,
                last_checked_time=normalized_time,
                last_checked_available=True
            )

            return {
                "requested_time": normalized_time,
                "available_at_requested_time": True,
                "message": f"The requested time {requested_time} is available on {date}"
            }

        # Step 3b: If NOT available, get alternatives
        else:
            logger.info(f"✗ Time {normalized_time} is not available for doctor {doctor_id} on {date} (session: {session_id})")
            logger.debug(f"Fetching alternative slots closest to {normalized_time}")

            # Get up to 4 closest alternative slots - pass all required parameters
            alternatives_str = self._get_alternative_slots(
                session_id=session_id,
                doctor_id=doctor_id,
                date=date,
                requested_time=normalized_time,
                available_timeslots=available_timeslots
            )

            logger.debug(f"Alternative slots response: {alternatives_str}")

            # Parse alternatives string into list
            alternative_slots = []
            if alternatives_str and alternatives_str != "No alternative slots available":
                # Split comma-separated string and clean
                alternative_slots = [s.strip() for s in alternatives_str.split(',') if s.strip()]
                logger.info(f"Found {len(alternative_slots)} alternative slots: {alternative_slots}")
            else:
                logger.warning(f"No alternative slots available for doctor {doctor_id} on {date}")

            # Update state
            self.state_manager.update_appointment_state(
                session_id,
                last_checked_time=normalized_time,
                last_checked_available=False
            )

            # Build message
            if alternative_slots:
                alt_list = ", ".join(alternative_slots)
                message = f"The requested time {normalized_time} is not available on {date}. Alternative times closest to your requested time: {alt_list}"
            else:
                message = f"The requested time {normalized_time} is not available on {date} and no alternative slots are available"

            return {
                "requested_time": normalized_time,
                "available_at_requested_time": False,
                "alternative_slots": alternative_slots,
                "message": message
            }

    def _calculate_availability_ranges(
        self,
        availability_windows: List[Dict[str, Any]],
        booked_slots: List[Dict[str, Any]]
    ) -> List[str]:
        """Calculate availability ranges by finding gaps between booked appointments.

        DEPRECATED: This method is kept for backward compatibility when availability
        windows are available. The preferred approach is _merge_timeslots_to_ranges()
        which uses the pre-filtered timeslots from the API.

        Example: If availability is 9:00-17:00 and booked slots are 10:00-10:30 and 15:00-15:30,
        returns ["9:00-10:00", "10:30-15:00", "15:30-17:00"]

        Args:
            availability_windows: List of availability windows with start_time and end_time
            booked_slots: List of booked slots with start_time and end_time

        Returns:
            List of availability ranges in "HH:MM-HH:MM" format
        """
        from datetime import datetime, timedelta

        if not availability_windows:
            return []

        ranges = []

        for window in availability_windows:
            window_start_str = window.get("start_time", "00:00:00")
            window_end_str = window.get("end_time", "23:59:59")

            # Parse window times
            window_start = datetime.strptime(window_start_str[:5], "%H:%M").time() if len(window_start_str) >= 5 else datetime.strptime("00:00", "%H:%M").time()
            window_end = datetime.strptime(window_end_str[:5], "%H:%M").time() if len(window_end_str) >= 5 else datetime.strptime("23:59", "%H:%M").time()

            # Get booked slots within this window
            window_booked = []
            for booked in booked_slots:
                booked_start_str = booked.get("start_time", "00:00:00")
                booked_end_str = booked.get("end_time", "23:59:59")
                booked_start = datetime.strptime(booked_start_str[:5], "%H:%M").time() if len(booked_start_str) >= 5 else datetime.strptime("00:00", "%H:%M").time()
                booked_end = datetime.strptime(booked_end_str[:5], "%H:%M").time() if len(booked_end_str) >= 5 else datetime.strptime("23:59", "%H:%M").time()

                # Check if booked slot overlaps with window
                if not (booked_end <= window_start or booked_start >= window_end):
                    window_booked.append({
                        "start": booked_start,
                        "end": booked_end
                    })

            # Sort booked slots by start time
            window_booked.sort(key=lambda x: x["start"])

            # Calculate ranges
            current_start = window_start

            for booked in window_booked:
                # If there's a gap before this booked slot, add it as a range
                if current_start < booked["start"]:
                    ranges.append(f"{current_start.strftime('%H:%M')}-{booked['start'].strftime('%H:%M')}")

                # Move current_start to after this booked slot
                current_start = booked["end"]

            # Add final range from last booked slot (or window start if no bookings) to window end
            if current_start < window_end:
                ranges.append(f"{current_start.strftime('%H:%M')}-{window_end.strftime('%H:%M')}")

        return ranges

    def tool_check_availability(
        self,
        session_id: str,
        doctor_id: str,
        date: str,
        requested_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check doctor availability with two distinct modes.

        MODE 1 - Range Mode (when requested_time is NOT provided):
            Returns availability_ranges - a list of continuous time blocks where doctor is available.
            Example: ["10:00-15:00", "15:30-16:00", "16:30-17:30"]

            The ranges represent continuous blocks of available time. Use these ranges to suggest
            appointment times to the patient.

        MODE 2 - Specific Time Mode (when requested_time IS provided):
            Checks if a specific time is available and returns alternatives if not.

            Example responses:
            - Available: {
                "requested_time": "14:00",
                "available_at_requested_time": True,
                "message": "The requested time 14:00 is available on 2025-12-05"
              }

            - Not available: {
                "requested_time": "14:00",
                "available_at_requested_time": False,
                "alternative_slots": ["14:30", "15:00", "13:30"],
                "message": "The requested time 14:00 is not available on 2025-12-05. Alternative times: 14:30, 15:00, 13:30"
              }

        Args:
            session_id: Session identifier
            doctor_id: Must be a valid UUID (use list_doctors first to get the ID)
            date: Date in YYYY-MM-DD format
            requested_time: Optional time to check. Supports multiple formats:
                - "14:00", "14:30" (24-hour format)
                - "2pm", "2:00pm", "2:00 PM" (12-hour with AM/PM)
                - "14", "02" (hour only)

        Returns:
            MODE 1 (no requested_time): {"availability_ranges": ["09:00-10:00", "10:30-15:00", ...]}
            MODE 2 (with requested_time): See examples above
        """
        import time as time_module
        start_time = time_module.time()
        
        logger.info("=" * 80)
        logger.info("APPOINTMENT_MANAGER: tool_check_availability() CALLED")
        logger.info("=" * 80)
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Input: doctor_id={doctor_id}, date={date}, requested_time={requested_time}")
        
        try:
            # Validate doctor_id is a UUID (not a name)
            import uuid
            try:
                uuid.UUID(doctor_id)
            except ValueError:
                logger.error(f"Invalid doctor_id format: {doctor_id} (session: {session_id})")
                error_response = {
                    "error": "Invalid doctor_id format",
                    "message": f"doctor_id must be a UUID, not a name. Got: {doctor_id}. Please use find_doctor_by_name first to get the correct doctor ID."
                }
                # Add appropriate fields based on mode
                if requested_time:
                    error_response["requested_time"] = requested_time
                    error_response["available_at_requested_time"] = False
                else:
                    error_response["availability_ranges"] = []
                return error_response

            # Get available timeslots (pre-filtered by API)
            logger.debug(f"Fetching available timeslots for doctor {doctor_id} on {date} (session: {session_id})")
            available_timeslots = self.db_client.get_available_time_slots(
                doctor_id=doctor_id,
                date=date,
                slot_duration_minutes=30
            )

            # Handle empty slots for both modes
            if not available_timeslots:
                logger.info(f"No available slots found for doctor {doctor_id} on {date} (session: {session_id})")
                if requested_time:
                    return {
                        "requested_time": requested_time,
                        "available_at_requested_time": False,
                        "alternative_slots": [],
                        "message": f"No available slots on {date}"
                    }
                else:
                    return {
                        "availability_ranges": [],
                        "message": f"No available slots on {date}"
                    }

            # MODE 1: Range Mode (existing behavior)
            if not requested_time or requested_time.strip() == "":
                logger.info(f"Range mode: returning availability ranges for doctor {doctor_id} on {date} (session: {session_id})")

                # Merge consecutive timeslots into ranges
                logger.debug(f"Using timeslots-based approach: {len(available_timeslots)} available slots")
                availability_ranges = self._merge_timeslots_to_ranges(available_timeslots)
                logger.info(f"Merged into {len(availability_ranges)} availability range(s): {availability_ranges}")

                # Update agent state
                self.state_manager.update_appointment_state(
                    session_id,
                    availability_ranges=availability_ranges
                )

                # Return ONLY availability_ranges - this is what the agent should use
                return {
                    "availability_ranges": availability_ranges
                }

            # MODE 2: Specific Time Mode (new behavior)
            else:
                logger.info(f"Specific time mode: checking availability for '{requested_time}' on {date} (session: {session_id})")
                return self._check_specific_time_availability(
                    session_id=session_id,
                    doctor_id=doctor_id,
                    date=date,
                    requested_time=requested_time,
                    available_timeslots=available_timeslots
                )

        except Exception as e:
            logger.error(f"Error checking availability for session {session_id}: {e}", exc_info=True)
            # Return appropriate format based on mode
            if requested_time:
                return {
                    "requested_time": requested_time,
                    "available_at_requested_time": False,
                    "error": str(e),
                    "message": f"An error occurred while checking availability: {str(e)}"
                }
            else:
                return {
                    "availability_ranges": [],
                    "error": str(e),
                    "message": f"An error occurred while checking availability: {str(e)}"
                }

    async def _execute_booking_workflow(
        self,
        session_id: str,
        patient_id: str,
        doctor_id: str,
        date: str,
        time: str,
        reason: str,
        correlation_id: str
    ) -> Dict[str, Any]:
        """
        Execute strict synchronous booking workflow with timeouts and verification.
        
        Workflow steps:
        1. Validate inputs
        2. Generate idempotency key
        3. Get clinic and appointment type
        4. Create appointment with idempotency key
        5. Final read verification from DB
        6. Return success only if verification passes
        
        Args:
            session_id: Session identifier
            patient_id: Patient ID
            doctor_id: Doctor ID
            date: Appointment date (YYYY-MM-DD)
            time: Appointment time (HH:MM)
            reason: Appointment reason
            correlation_id: Correlation ID for logging
        
        Returns:
            Dict with success status and appointment details or error
        """
        import time as time_module
        import asyncio
        from datetime import datetime as dt
        
        workflow_start = time_module.time()
        logger.info(f"🚀 BOOKING WORKFLOW START: correlation_id={correlation_id}")
        logger.info(f"   patient_id={patient_id}, doctor_id={doctor_id}, date={date}, time={time}")
        
        try:
            # Step 1: Validate patient_id
            if not patient_id or patient_id.strip() == "":
                logger.error(f"❌ Validation failed: patient_id is empty - correlation_id={correlation_id}")
                return {
                    "success": False,
                    "error": "Patient registration required",
                    "error_code": "VALIDATION_FAILED",
                    "message": "You must complete registration before booking an appointment. Please register first."
                }
            
            # Step 2: Generate idempotency key
            idempotency_key = self._generate_idempotency_key(patient_id, doctor_id, date, time, reason)
            logger.info(f"🔑 Idempotency key: {idempotency_key} - correlation_id={correlation_id}")
            
            # Step 3: Get clinic ID
            clinic_info = self.db_client.get_clinic_info()
            if not clinic_info:
                all_clinics = self.db_client.get_all_clinics()
                if all_clinics and len(all_clinics) > 0:
                    clinic_id = all_clinics[0].get("id")
                    logger.info(f"Using first available clinic: {clinic_id} - correlation_id={correlation_id}")
                else:
                    clinic_id = "11111111-1111-1111-1111-111111111111"
                    logger.warning(f"No clinic found, using default: {clinic_id} - correlation_id={correlation_id}")
            else:
                clinic_id = clinic_info.get("id")
                logger.info(f"Using clinic: {clinic_id} - correlation_id={correlation_id}")
            
            # Step 4: Get appointment type
            appointment_types = self.db_client.get_appointment_types()
            appointment_type_id = appointment_types[0].get("id") if appointment_types else None
            
            if not appointment_type_id:
                logger.error(f"❌ Appointment type not found - correlation_id={correlation_id}")
                return {
                    "success": False,
                    "error": "Appointment type not found",
                    "error_code": "APPOINTMENT_TYPE_NOT_FOUND",
                    "message": "Unable to determine appointment type. Please try again."
                }
            
            # Step 5: Calculate end time
            start_dt = dt.strptime(time, "%H:%M")
            end_dt = start_dt + timedelta(minutes=30)
            end_time = end_dt.strftime("%H:%M")
            
            # Step 6: Create appointment with idempotency key (with timeout and retry)
            logger.info(f"📋 Creating appointment with idempotency_key={idempotency_key} - correlation_id={correlation_id}")
            create_start = time_module.time()
            
            def create_appointment_sync():
                return self.db_client.create_appointment(
                    clinic_id=clinic_id,
                    patient_id=patient_id,
                    doctor_id=doctor_id,
                    appointment_type_id=appointment_type_id,
                    appointment_date=date,
                    start_time=time,
                    end_time=end_time,
                    reason=reason,
                    emergency_level="routine",
                    request_id=idempotency_key  # Pass idempotency key as request_id
                )
            
            try:
                appointment = await self._execute_with_timeout(
                    create_appointment_sync,
                    timeout_seconds=10.0,
                    max_retries=2,
                    correlation_id=correlation_id
                )
                create_duration_ms = (time_module.time() - create_start) * 1000
                logger.info(f"📊 Create appointment completed in {create_duration_ms:.0f}ms - correlation_id={correlation_id}")
            except TimeoutError:
                logger.error(f"❌ Create appointment timed out - correlation_id={correlation_id}")
                return {
                    "success": False,
                    "error": "Create appointment timed out",
                    "error_code": "TIMEOUT_CREATE",
                    "message": "Sorry — I couldn't schedule your appointment right now because of a system error. Would you like me to try again? (Yes / No)"
                }
            except Exception as e:
                error_str = str(e).lower()
                if "409" in error_str or "conflict" in error_str:
                    logger.warning(f"⚠️ Slot conflict detected - correlation_id={correlation_id}")
                    # Get alternatives for user
                    alternatives = self._get_alternative_slots(doctor_id, date, time)
                    return {
                        "success": False,
                        "error": "Slot unavailable",
                        "error_code": "SLOT_CONFLICT",
                        "message": f"I couldn't book that time — Dr. [name] is not available at {date} {time}. Would you like these alternatives: {alternatives}?"
                    }
                else:
                    logger.error(f"❌ Create appointment failed: {e} - correlation_id={correlation_id}")
                    return {
                        "success": False,
                        "error": str(e),
                        "error_code": "CREATE_FAILED",
                        "message": "Sorry — I couldn't schedule your appointment right now because of a system error. Would you like me to try again? (Yes / No)"
                    }
            
            if not appointment:
                logger.error(f"❌ Create appointment returned None - correlation_id={correlation_id}")
                return {
                    "success": False,
                    "error": "Create appointment returned None",
                    "error_code": "CREATE_RETURNED_NONE",
                    "message": "Sorry — I couldn't schedule your appointment right now because of a system error. Would you like me to try again? (Yes / No)"
                }
            
            appointment_id = appointment.get('id')
            logger.info(f"✅ Appointment created: {appointment_id} - correlation_id={correlation_id}")
            
            # Step 7: Final read verification (CRITICAL - never skip this)
            logger.info(f"🔍 FINAL READ VERIFY: Starting verification - correlation_id={correlation_id}")
            verify_start = time_module.time()
            
            # Final read verify is synchronous, but we wrap it for timeout
            try:
                verified_appointment = await self._execute_with_timeout(
                    lambda: self._final_read_verify(
                        appointment_id=appointment_id,
                        patient_id=patient_id,
                        doctor_id=doctor_id,
                        date=date,
                        time=time,
                        correlation_id=correlation_id
                    ),
                    timeout_seconds=10.0,
                    max_retries=2,
                    correlation_id=correlation_id
                )
                verify_duration_ms = (time_module.time() - verify_start) * 1000
                logger.info(f"📊 Final read verify completed in {verify_duration_ms:.0f}ms - correlation_id={correlation_id}")
            except TimeoutError:
                logger.error(f"❌ Final read verify timed out - correlation_id={correlation_id}")
                # CRITICAL: Do NOT confirm if verification times out
                return {
                    "success": False,
                    "error": "Final read verify timed out",
                    "error_code": "TIMEOUT_VERIFY",
                    "message": "I tried to confirm the booking but couldn't verify it in our system. I did not book the appointment. Would you like me to try again?"
                }
            except Exception as e:
                logger.error(f"❌ Final read verify failed: {e} - correlation_id={correlation_id}")
                return {
                    "success": False,
                    "error": "Final read verify failed",
                    "error_code": "VERIFY_FAILED",
                    "message": "I tried to confirm the booking but couldn't verify it in our system. I did not book the appointment. Would you like me to try again?"
                }
            
            if not verified_appointment:
                # CRITICAL: Verification failed - do NOT confirm
                logger.error(f"❌ FINAL READ VERIFY FAILED: Appointment not found in DB - correlation_id={correlation_id}")
                logger.error(f"   🚨 FALSE CONFIRMATION PREVENTED - NOT confirming to user")
                
                # Log metrics
                workflow_duration_ms = (time_module.time() - workflow_start) * 1000
                self._log_booking_metrics(
                    appointment_inserted=False,
                    db_row_id=appointment_id,
                    verification_latency_ms=verify_duration_ms,
                    scheduling_latency_ms=create_duration_ms,
                    dashboard_visible=False,
                    request_id=idempotency_key,
                    mismatch_detected=True,
                    error_code="FINAL_VERIFY_FAILED"
                )
                
                logger.info(f"📊 METRIC: appointment_booking.failure correlation_id={correlation_id} latency_ms={workflow_duration_ms:.0f}")
                
                return {
                    "success": False,
                    "error": "Final read verify failed - appointment not in DB",
                    "error_code": "FINAL_VERIFY_FAILED",
                    "message": "I tried to confirm the booking but couldn't verify it in our system. I did not book the appointment. Would you like me to try again?"
                }
            
            # Step 8: SUCCESS - Verification passed
            logger.info(f"✅ FINAL READ VERIFY PASSED: Appointment confirmed in DB - correlation_id={correlation_id}")
            
            # Get doctor name for success message
            doctor = self.db_client.get_doctor_by_id(doctor_id)
            doctor_name = f"Dr. {doctor.get('first_name', '')} {doctor.get('last_name', '')}" if doctor else "Dr. [name]"
            
            # Format date for display
            try:
                date_obj = dt.strptime(date, "%Y-%m-%d")
                date_display = date_obj.strftime("%B %d, %Y")
            except:
                date_display = date
            
            # Format time for display
            try:
                time_obj = dt.strptime(time, "%H:%M")
                time_display = time_obj.strftime("%I:%M %p")
            except:
                time_display = time
            
            # Log success metrics
            workflow_duration_ms = (time_module.time() - workflow_start) * 1000
            self._log_booking_metrics(
                appointment_inserted=True,
                db_row_id=appointment_id,
                verification_latency_ms=verify_duration_ms,
                scheduling_latency_ms=create_duration_ms,
                dashboard_visible=True,
                request_id=idempotency_key,
                mismatch_detected=False,
                error_code=None
            )
            
            logger.info(f"📊 METRIC: appointment_booking.success correlation_id={correlation_id} latency_ms={workflow_duration_ms:.0f}")
            
            # Update state
            self.state_manager.update_appointment_state(
                session_id,
                workflow_step="completed",
                operation_type="booking"
            )
            
            # Return success with formatted message
            return {
                "success": True,
                "appointment": verified_appointment,
                "appointment_id": appointment_id,
                "verified": True,
                "idempotency_key": idempotency_key,
                "verification_latency_ms": round(verify_duration_ms, 2),
                "workflow_latency_ms": round(workflow_duration_ms, 2),
                "message": f"✅ Your appointment is confirmed.\n\n**Doctor:** {doctor_name}\n**Date:** {date_display}\n**Time:** {time_display}\n**Reason:** {reason}\n**Appointment ID:** {appointment_id}"
            }
            
        except Exception as e:
            workflow_duration_ms = (time_module.time() - workflow_start) * 1000
            logger.error(f"❌ WORKFLOW EXCEPTION: {e} - correlation_id={correlation_id}", exc_info=True)
            logger.info(f"📊 METRIC: appointment_booking.failure correlation_id={correlation_id} latency_ms={workflow_duration_ms:.0f}")
            
            return {
                "success": False,
                "error": str(e),
                "error_code": "WORKFLOW_EXCEPTION",
                "message": "Sorry — I couldn't schedule your appointment right now because of a system error. Would you like me to try again? (Yes / No)"
            }
    
    
    
    def tool_book_appointment(
        self,
        session_id: str,
        patient_id: str,
        doctor_id: str,
        date: str,
        time: str,
        reason: str
    ) -> Dict[str, Any]:
        """
        Book a new appointment using strict synchronous workflow.
        
        This method orchestrates the complete booking flow with timeouts,
        retries, idempotency, and mandatory final DB verification.
        """
        import asyncio
        import time as time_module
        
        correlation_id = session_id
        overall_start = time_module.time()
        
        logger.info("=" * 80)
        logger.info("APPOINTMENT_MANAGER: tool_book_appointment() CALLED")
        logger.info("=" * 80)
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Input: patient_id={patient_id}, doctor_id={doctor_id}, date={date}, time={time}, reason={reason}")
        logger.info(f"📋 BOOKING REQUEST: correlation_id={correlation_id}")
        
        try:
            # Execute workflow with overall timeout (20 seconds)
            # Handle both sync and async contexts
            try:
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                if loop.is_running():
                    # If loop is already running, we need to use a different approach
                    # Create a task and wait for it
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            lambda: asyncio.run(
                                asyncio.wait_for(
                                    self._execute_booking_workflow(
                                        session_id=session_id,
                                        patient_id=patient_id,
                                        doctor_id=doctor_id,
                                        date=date,
                                        time=time,
                                        reason=reason,
                                        correlation_id=correlation_id
                                    ),
                                    timeout=20.0
                                )
                            )
                        )
                        result = future.result(timeout=25.0)  # Slightly longer than async timeout
                else:
                    result = loop.run_until_complete(
                        asyncio.wait_for(
                            self._execute_booking_workflow(
                                session_id=session_id,
                                patient_id=patient_id,
                                doctor_id=doctor_id,
                                date=date,
                                time=time,
                                reason=reason,
                                correlation_id=correlation_id
                            ),
                            timeout=20.0  # Overall orchestration timeout
                        )
                    )
            except (asyncio.TimeoutError, concurrent.futures.TimeoutError):
                overall_duration_ms = (time_module.time() - overall_start) * 1000
                logger.error(f"❌ OVERALL TIMEOUT: Workflow exceeded 20s - correlation_id={correlation_id}")
                logger.info(f"📊 METRIC: appointment_booking.failure correlation_id={correlation_id} latency_ms={overall_duration_ms:.0f} error_code=OVERALL_TIMEOUT")
                
                result = {
                    "success": False,
                    "error": "Overall workflow timeout",
                    "error_code": "OVERALL_TIMEOUT",
                    "message": "Sorry — I couldn't schedule your appointment right now because of a system error. Would you like me to try again? (Yes / No)"
                }
            
            overall_duration_ms = (time_module.time() - overall_start) * 1000
            logger.info(f"📊 Overall workflow completed in {overall_duration_ms:.0f}ms - correlation_id={correlation_id}")
            logger.info(f"Output: success={result.get('success')}, appointment_id={result.get('appointment_id', 'N/A')}")
            logger.info(f"Time taken: {overall_duration_ms:.2f}ms")
            logger.info("=" * 80)
            
            return result
            
        except Exception as e:
            overall_duration_ms = (time_module.time() - overall_start) * 1000
            logger.error(f"❌ BOOKING EXCEPTION: {e} - correlation_id={correlation_id}", exc_info=True)
            logger.info(f"📊 METRIC: appointment_booking.failure correlation_id={correlation_id} latency_ms={overall_duration_ms:.0f}")
            
            result = {
                "success": False,
                "error": str(e),
                "error_code": "BOOKING_EXCEPTION",
                "message": "Sorry — I couldn't schedule your appointment right now because of a system error. Would you like me to try again? (Yes / No)"
            }
            
            return result

    async def _rollback_appointments(
        self,
        session_id: str,
        appointment_ids: List[str],
        correlation_id: str
    ) -> Dict[str, Any]:
        """
        Rollback (cancel) previously booked appointments.

        Used when a multi-booking operation partially fails.

        Args:
            session_id: Session identifier
            appointment_ids: List of appointment IDs to cancel
            correlation_id: Correlation ID for logging

        Returns:
            Dict with rollback results
        """
        logger.info(f"🔄 ROLLBACK: Cancelling {len(appointment_ids)} appointments - correlation_id={correlation_id}")

        rollback_results = []
        for apt_id in appointment_ids:
            try:
                result = self.db_client.cancel_appointment(
                    appointment_id=apt_id,
                    reason="Rollback due to partial multi-booking failure"
                )
                rollback_results.append({
                    "appointment_id": apt_id,
                    "cancelled": True
                })
                logger.info(f"✅ Rolled back appointment {apt_id}")
            except Exception as e:
                rollback_results.append({
                    "appointment_id": apt_id,
                    "cancelled": False,
                    "error": str(e)
                })
                logger.error(f"❌ Failed to rollback appointment {apt_id}: {e}")

        return {
            "rollback_attempted": len(appointment_ids),
            "rollback_succeeded": sum(1 for r in rollback_results if r.get("cancelled")),
            "results": rollback_results
        }

    def tool_book_multiple_appointments(
        self,
        session_id: str,
        patient_id: str,
        appointments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Book multiple appointments at once with different parameters.

        This method books multiple appointments sequentially, each with its own
        doctor, date, time, and reason. All appointments are for the same patient.

        Args:
            session_id: Session identifier
            patient_id: Patient's ID (same for all appointments)
            appointments: List of appointment dicts, each containing:
                - doctor_id: Doctor's UUID
                - date: Appointment date (YYYY-MM-DD)
                - time: Appointment time (HH:MM)
                - reason: Appointment reason

        Returns:
            Dict with:
            - success: True if all bookings succeeded, False if any failed
            - results: List of individual booking results
            - summary: Human-readable summary
            - successful_count: Number of successful bookings
            - failed_count: Number of failed bookings
        """
        import asyncio
        import time as time_module

        overall_start = time_module.time()
        correlation_id = session_id

        logger.info("=" * 80)
        logger.info("APPOINTMENT_MANAGER: tool_book_multiple_appointments() CALLED")
        logger.info("=" * 80)
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Input: patient_id={patient_id}, appointments_count={len(appointments)}")
        logger.info(f"📋 MULTIPLE BOOKING REQUEST: correlation_id={correlation_id}")

        # Validate inputs
        if not patient_id or patient_id.strip() == "":
            return {
                "success": False,
                "error": "Patient registration required",
                "message": "You must complete registration before booking appointments. Please register first.",
                "results": []
            }

        if not appointments or len(appointments) == 0:
            return {
                "success": False,
                "error": "No appointments provided",
                "message": "Please provide at least one appointment to book.",
                "results": []
            }

        # Validate each appointment has required fields
        for idx, apt in enumerate(appointments):
            if not all(k in apt for k in ['doctor_id', 'date', 'time', 'reason']):
                return {
                    "success": False,
                    "error": f"Appointment {idx + 1} missing required fields",
                    "message": f"Appointment {idx + 1} is missing required fields (doctor_id, date, time, reason)",
                    "results": []
                }

        # Book each appointment sequentially
        results = []
        successful_bookings = []
        failed_bookings = []

        for idx, apt in enumerate(appointments):
            logger.info(f"📋 Booking appointment {idx + 1}/{len(appointments)}")
            logger.info(f"   doctor_id={apt['doctor_id']}, date={apt['date']}, time={apt['time']}, reason={apt['reason']}")

            try:
                # Execute booking workflow for this appointment
                try:
                    loop = None
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    if loop.is_running():
                        # If loop is already running, use ThreadPoolExecutor
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                lambda: asyncio.run(
                                    asyncio.wait_for(
                                        self._execute_booking_workflow(
                                            session_id=session_id,
                                            patient_id=patient_id,
                                            doctor_id=apt['doctor_id'],
                                            date=apt['date'],
                                            time=apt['time'],
                                            reason=apt['reason'],
                                            correlation_id=f"{correlation_id}_apt{idx + 1}"
                                        ),
                                        timeout=20.0
                                    )
                                )
                            )
                            result = future.result(timeout=25.0)
                    else:
                        result = loop.run_until_complete(
                            asyncio.wait_for(
                                self._execute_booking_workflow(
                                    session_id=session_id,
                                    patient_id=patient_id,
                                    doctor_id=apt['doctor_id'],
                                    date=apt['date'],
                                    time=apt['time'],
                                    reason=apt['reason'],
                                    correlation_id=f"{correlation_id}_apt{idx + 1}"
                                ),
                                timeout=20.0
                            )
                        )
                except (asyncio.TimeoutError, concurrent.futures.TimeoutError):
                    logger.error(f"❌ Appointment {idx + 1} timed out")
                    result = {
                        "success": False,
                        "error": "Timeout",
                        "error_code": "TIMEOUT",
                        "message": f"Appointment {idx + 1} timed out",
                        "appointment_number": idx + 1
                    }

                # Add appointment details to result
                result['appointment_number'] = idx + 1
                result['requested_doctor_id'] = apt['doctor_id']
                result['requested_date'] = apt['date']
                result['requested_time'] = apt['time']
                result['requested_reason'] = apt['reason']

                results.append(result)

                if result.get('success'):
                    successful_bookings.append(result)
                    logger.info(f"✅ Appointment {idx + 1} booked successfully: {result.get('appointment_id')}")
                else:
                    failed_bookings.append(result)
                    logger.error(f"❌ Appointment {idx + 1} failed: {result.get('error')}")

            except Exception as e:
                logger.error(f"❌ Exception booking appointment {idx + 1}: {e}", exc_info=True)
                error_result = {
                    "success": False,
                    "error": str(e),
                    "error_code": "EXCEPTION",
                    "message": f"Appointment {idx + 1} failed with exception",
                    "appointment_number": idx + 1,
                    "requested_doctor_id": apt['doctor_id'],
                    "requested_date": apt['date'],
                    "requested_time": apt['time'],
                    "requested_reason": apt['reason']
                }
                results.append(error_result)
                failed_bookings.append(error_result)

        # Calculate overall duration
        overall_duration_ms = (time_module.time() - overall_start) * 1000

        # Generate summary
        all_successful = len(failed_bookings) == 0
        successful_count = len(successful_bookings)
        failed_count = len(failed_bookings)

        logger.info(f"📊 Multiple booking completed in {overall_duration_ms:.0f}ms")
        logger.info(f"   Successful: {successful_count}/{len(appointments)}")
        logger.info(f"   Failed: {failed_count}/{len(appointments)}")
        logger.info(f"Output: success={all_successful}, successful_count={successful_count}, failed_count={failed_count}")
        logger.info(f"Time taken: {overall_duration_ms:.2f}ms")

        # PHASE 4: Rollback on partial failure
        rollback_result = None
        if failed_count > 0 and successful_count > 0:
            logger.warning(
                f"⚠️ PARTIAL FAILURE: {successful_count} succeeded, {failed_count} failed - correlation_id={correlation_id}"
            )

            # Perform automatic rollback for consistency
            successful_ids = [r.get("appointment_id") for r in successful_bookings if r.get("appointment_id")]
            try:
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                if loop.is_running():
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            lambda: asyncio.run(
                                self._rollback_appointments(
                                    session_id=session_id,
                                    appointment_ids=successful_ids,
                                    correlation_id=correlation_id
                                )
                            )
                        )
                        rollback_result = future.result(timeout=10.0)
                else:
                    rollback_result = loop.run_until_complete(
                        self._rollback_appointments(
                            session_id=session_id,
                            appointment_ids=successful_ids,
                            correlation_id=correlation_id
                        )
                    )
            except Exception as e:
                logger.error(f"❌ Rollback failed: {e}")
                rollback_result = {"error": str(e)}

            return {
                "success": False,
                "error": "Partial booking failure - all bookings rolled back for consistency",
                "message": (
                    f"I wasn't able to book all {len(appointments)} appointments. "
                    f"To keep things consistent, I've cancelled the ones that did go through. "
                    f"Would you like me to try booking them again one at a time?"
                ),
                "results": results,
                "rollback": rollback_result,
                "successful_count": 0,  # After rollback
                "failed_count": len(appointments),
                "total_count": len(appointments),
                "overall_duration_ms": round(overall_duration_ms, 2)
            }

        # PHASE 4: Rollback on partial failure
        rollback_result = None
        if failed_count > 0 and successful_count > 0:
            logger.warning(
                f"⚠️ PARTIAL FAILURE: {successful_count} succeeded, {failed_count} failed - correlation_id={correlation_id}"
            )

            # Perform automatic rollback for consistency
            successful_ids = [r.get("appointment_id") for r in successful_bookings if r.get("appointment_id")]
            try:
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                if loop.is_running():
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            lambda: asyncio.run(
                                self._rollback_appointments(
                                    session_id=session_id,
                                    appointment_ids=successful_ids,
                                    correlation_id=correlation_id
                                )
                            )
                        )
                        rollback_result = future.result(timeout=10.0)
                else:
                    rollback_result = loop.run_until_complete(
                        self._rollback_appointments(
                            session_id=session_id,
                            appointment_ids=successful_ids,
                            correlation_id=correlation_id
                        )
                    )
            except Exception as e:
                logger.error(f"❌ Rollback failed: {e}")
                rollback_result = {"error": str(e)}

            return {
                "success": False,
                "error": "Partial booking failure - all bookings rolled back for consistency",
                "message": (
                    f"I wasn't able to book all {len(appointments)} appointments. "
                    f"To keep things consistent, I've cancelled the ones that did go through. "
                    f"Would you like me to try booking them again one at a time?"
                ),
                "results": results,
                "rollback": rollback_result,
                "successful_count": 0,  # After rollback
                "failed_count": len(appointments),
                "total_count": len(appointments),
                "overall_duration_ms": round(overall_duration_ms, 2)
            }

        # Build formatted message
        if all_successful:
            # Format success message
            message_lines = [f"✅ All {successful_count} appointments confirmed:"]
            for result in successful_bookings:
                apt_id = result.get('appointment_id', 'N/A')
                date = result.get('requested_date', 'N/A')
                time = result.get('requested_time', 'N/A')
                reason = result.get('requested_reason', 'N/A')

                # Format date
                try:
                    date_obj = datetime.strptime(date, "%Y-%m-%d")
                    date_display = date_obj.strftime("%B %d, %Y")
                except:
                    date_display = date

                # Format time
                try:
                    time_obj = datetime.strptime(time, "%H:%M")
                    time_display = time_obj.strftime("%I:%M %p")
                except:
                    time_display = time

                message_lines.append(f"{result['appointment_number']}) {reason} - {date_display} at {time_display} (ID: {apt_id})")

            message = "\n".join(message_lines)
        else:
            # Partial success or complete failure
            message_lines = []

            if successful_count > 0:
                message_lines.append(f"⚠️ Booked {successful_count} of {len(appointments)} appointments:")
                for result in successful_bookings:
                    apt_id = result.get('appointment_id', 'N/A')
                    reason = result.get('requested_reason', 'N/A')
                    message_lines.append(f"✅ {result['appointment_number']}) {reason} (ID: {apt_id})")

            if failed_count > 0:
                message_lines.append(f"\n❌ Failed to book {failed_count} appointments:")
                for result in failed_bookings:
                    reason = result.get('requested_reason', 'N/A')
                    error = result.get('error', 'Unknown error')
                    message_lines.append(f"   {result['appointment_number']}) {reason} - {error}")

            message = "\n".join(message_lines)

        # Update state
        if all_successful:
            self.state_manager.update_appointment_state(
                session_id,
                workflow_step="completed",
                operation_type="multiple_booking"
            )

        result = {
            "success": all_successful,
            "results": results,
            "successful_count": successful_count,
            "failed_count": failed_count,
            "total_count": len(appointments),
            "overall_duration_ms": round(overall_duration_ms, 2),
            "message": message
        }
        
        logger.info("=" * 80)
        
        return result

    def tool_check_patient_appointments(
        self,
        session_id: str,
        patient_id: str
    ) -> Dict[str, Any]:
        """Get patient's appointments.""" 
        import time as time_module
        start_time = time_module.time()
        
        logger.info("=" * 80)
        logger.info("APPOINTMENT_MANAGER: tool_check_patient_appointments() CALLED")
        logger.info("=" * 80)
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Input: patient_id={patient_id}")
        
        try:
            appointments = self.db_client.get_patient_appointments(patient_id)

            if not appointments:
                duration_ms = (time_module.time() - start_time) * 1000
                result = {
                    "success": True,
                    "appointments": [],
                    "message": "No appointments found"
                }
                logger.info(f"Output: {result}")
                logger.info(f"Time taken: {duration_ms:.2f}ms")
                logger.info("=" * 80)
                return result

            duration_ms = (time_module.time() - start_time) * 1000
            result = {
                "success": True,
                "appointments": appointments,
                "count": len(appointments)
            }
            
            logger.info(f"Output: success=True, count={len(appointments)}")
            logger.info(f"Time taken: {duration_ms:.2f}ms")
            logger.info("=" * 80)
            
            return result

        except Exception as e:
            duration_ms = (time_module.time() - start_time) * 1000
            logger.error(f"Error fetching appointments: {e}")
            logger.info(f"Output: {{'error': '{str(e)}'}}")
            logger.info(f"Time taken: {duration_ms:.2f}ms")
            logger.info("=" * 80)
            return {"error": str(e)}

    def tool_cancel_appointment(
        self,
        session_id: str,
        appointment_id: str,
        reason: str
    ) -> Dict[str, Any]:
        """Cancel an appointment."""
        import time as time_module
        start_time = time_module.time()
        
        logger.info("=" * 80)
        logger.info("APPOINTMENT_MANAGER: tool_cancel_appointment() CALLED")
        logger.info("=" * 80)
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Input: appointment_id={appointment_id}, reason={reason}")
        
        try:
            result = self.db_client.cancel_appointment(appointment_id, reason)

            if result:
                duration_ms = (time_module.time() - start_time) * 1000
                output = {
                    "success": True,
                    "message": f"Appointment {appointment_id} cancelled successfully"
                }
                logger.info(f"Output: {output}")
                logger.info(f"Time taken: {duration_ms:.2f}ms")
                logger.info("=" * 80)
                return output
            else:
                duration_ms = (time_module.time() - start_time) * 1000
                output = {"error": "Failed to cancel appointment"}
                logger.info(f"Output: {output}")
                logger.info(f"Time taken: {duration_ms:.2f}ms")
                logger.info("=" * 80)
                return output

        except Exception as e:
            duration_ms = (time_module.time() - start_time) * 1000
            logger.error(f"Error cancelling appointment: {e}")
            logger.info(f"Output: {{'error': '{str(e)}'}}")
            logger.info(f"Time taken: {duration_ms:.2f}ms")
            logger.info("=" * 80)
            return {"error": str(e)}

    def tool_reschedule_appointment(
        self,
        session_id: str,
        appointment_id: str,
        new_date: str,
        new_time: str
    ) -> Dict[str, Any]:
        """Reschedule an appointment."""
        import time as time_module
        start_time = time_module.time()
        
        logger.info("=" * 80)
        logger.info("APPOINTMENT_MANAGER: tool_reschedule_appointment() CALLED")
        logger.info("=" * 80)
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Input: appointment_id={appointment_id}, new_date={new_date}, new_time={new_time}")
        
        try:
            # Calculate new end time
            start_dt = datetime.strptime(new_time, "%H:%M")
            end_dt = start_dt + timedelta(minutes=30)
            new_end_time = end_dt.strftime("%H:%M")

            result = self.db_client.reschedule_appointment(
                appointment_id,
                new_date,
                new_time,
                new_end_time
            )

            if result:
                duration_ms = (time_module.time() - start_time) * 1000
                output = {
                    "success": True,
                    "appointment": result,
                    "message": f"Appointment rescheduled to {new_date} at {new_time}"
                }
                logger.info(f"Output: success=True, appointment_id={result.get('id', 'N/A')}")
                logger.info(f"Time taken: {duration_ms:.2f}ms")
                logger.info("=" * 80)
                return output
            else:
                duration_ms = (time_module.time() - start_time) * 1000
                output = {"error": "Failed to reschedule appointment"}
                logger.info(f"Output: {output}")
                logger.info(f"Time taken: {duration_ms:.2f}ms")
                logger.info("=" * 80)
                return output

        except Exception as e:
            duration_ms = (time_module.time() - start_time) * 1000
            logger.error(f"Error rescheduling appointment: {e}")
            logger.info(f"Output: {{'error': '{str(e)}'}}")
            logger.info(f"Time taken: {duration_ms:.2f}ms")
            logger.info("=" * 80)
            return {"error": str(e)}

    def tool_update_appointment(
        self,
        session_id: str,
        appointment_id: str,
        doctor_id: Optional[str] = None,
        clinic_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        appointment_type_id: Optional[str] = None,
        appointment_date: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        status: Optional[str] = None,
        reason: Optional[str] = None,
        notes: Optional[str] = None,
        emergency_level: Optional[str] = None,
        follow_up_required: Optional[bool] = None,
        follow_up_days: Optional[int] = None,
        procedure_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update an appointment with one or more parameters.
        
        This tool allows updating any combination of appointment fields in a single call.
        Only the fields provided will be updated; all other fields remain unchanged.
        """
        try:
            # Build updates dictionary - only include non-None values
            updates = {}
            
            if doctor_id is not None:
                updates["doctor_id"] = doctor_id
            if clinic_id is not None:
                updates["clinic_id"] = clinic_id
            if patient_id is not None:
                updates["patient_id"] = patient_id
            if appointment_type_id is not None:
                updates["appointment_type_id"] = appointment_type_id
            if appointment_date is not None:
                updates["appointment_date"] = appointment_date
            if start_time is not None:
                updates["start_time"] = start_time
                # If start_time is provided but end_time is not, calculate it (30 min default)
                if end_time is None:
                    try:
                        start_dt = datetime.strptime(start_time, "%H:%M")
                        end_dt = start_dt + timedelta(minutes=30)
                        updates["end_time"] = end_dt.strftime("%H:%M")
                        logger.info(f"Calculated end_time: {updates['end_time']} from start_time: {start_time}")
                    except ValueError as e:
                        logger.warning(f"Could not parse start_time '{start_time}' to calculate end_time: {e}")
            if end_time is not None:
                updates["end_time"] = end_time
            if status is not None:
                # Validate status
                valid_statuses = ['scheduled', 'confirmed', 'checked_in', 'in_progress', 'completed', 'cancelled', 'no_show', 'rescheduled']
                if status not in valid_statuses:
                    return {
                        "success": False,
                        "error": f"Invalid status '{status}'. Valid values: {', '.join(valid_statuses)}"
                    }
                updates["status"] = status
            if reason is not None:
                updates["reason"] = reason
            if notes is not None:
                updates["notes"] = notes
            if emergency_level is not None:
                # Validate emergency_level
                valid_levels = ['routine', 'urgent', 'emergency', 'critical']
                if emergency_level not in valid_levels:
                    return {
                        "success": False,
                        "error": f"Invalid emergency_level '{emergency_level}'. Valid values: {', '.join(valid_levels)}"
                    }
                updates["emergency_level"] = emergency_level
            if follow_up_required is not None:
                updates["follow_up_required"] = follow_up_required
            if follow_up_days is not None:
                updates["follow_up_days"] = follow_up_days
            if procedure_type is not None:
                updates["procedure_type"] = procedure_type
            
            # Check if at least one field is being updated
            if not updates:
                return {
                    "success": False,
                    "error": "No fields provided to update. Please specify at least one field to update."
                }
            
            logger.info(f"Updating appointment {appointment_id} with fields: {list(updates.keys())}")
            
            # Call the database client update method
            result = self.db_client.update_appointment(appointment_id, updates)
            
            if result:
                # Format success message with what was updated
                updated_fields = list(updates.keys())
                return {
                    "success": True,
                    "appointment": result,
                    "updated_fields": updated_fields,
                    "message": f"Appointment updated successfully. Updated fields: {', '.join(updated_fields)}"
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to update appointment. The appointment may not exist or the update may have failed."
                }
                
        except ValueError as e:
            logger.error(f"Value error updating appointment: {e}")
            return {
                "success": False,
                "error": f"Invalid input format: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Error updating appointment: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    # ========== SYNCHRONOUS BOOKING WORKFLOW INFRASTRUCTURE ==========
    # These methods support the strict synchronous booking workflow
    
    def _generate_idempotency_key(
        self,
        patient_id: str,
        doctor_id: str,
        date: str,
        time: str,
        reason: str
    ) -> str:
        """
        Generate idempotency key using SHA256 hash.
        
        Args:
            patient_id: Patient ID
            doctor_id: Doctor ID
            date: Appointment date (YYYY-MM-DD)
            time: Appointment time (HH:MM)
            reason: Appointment reason
        
        Returns:
            32-character hex digest of SHA256 hash
        """
        import hashlib
        
        key_string = f"{patient_id}_{doctor_id}_{date}_{time}_{reason}"
        idempotency_key = hashlib.sha256(key_string.encode()).hexdigest()[:32]
        
        logger.info(f"🔑 Generated idempotency_key: {idempotency_key} (from: {key_string[:50]}...)")
        return idempotency_key
    
    def _is_transient_error(self, error: Exception) -> bool:
        """
        Determine if an error is transient and should be retried.
        
        Args:
            error: The exception that occurred
        
        Returns:
            True if error is transient (5xx, timeout), False otherwise
        """
        error_str = str(error).lower()
        
        # Check for HTTP status codes in error message
        if "500" in error_str or "502" in error_str or "503" in error_str or "504" in error_str:
            return True
        
        # Check for timeout errors
        if "timeout" in error_str or "timed out" in error_str:
            return True
        
        # Check for network errors
        if "connection" in error_str or "network" in error_str:
            return True
        
        # Do NOT retry on these errors
        if "409" in error_str or "conflict" in error_str:
            return False
        if "404" in error_str or "not found" in error_str:
            return False
        if "400" in error_str or "bad request" in error_str:
            return False
        
        # Default: treat as transient if it's a generic exception
        return True
    
    async def _execute_with_timeout(
        self,
        func,
        timeout_seconds: float = 10.0,
        max_retries: int = 2,
        correlation_id: Optional[str] = None
    ) -> Any:
        """
        Execute a function with timeout and retry logic.
        
        Args:
            func: Async function to execute
            timeout_seconds: Timeout per attempt (default 10s)
            max_retries: Maximum retry attempts (default 2, so 3 total attempts)
            correlation_id: Session ID for logging
        
        Returns:
            Function result
        
        Raises:
            TimeoutError: If all attempts timeout
            Exception: If all retries fail
        """
        import asyncio
        import time as time_module
        
        correlation_id = correlation_id or "unknown"
        last_error = None
        
        for attempt in range(max_retries + 1):  # +1 because first attempt is not a retry
            try:
                logger.info(f"⏱️ Attempt {attempt + 1}/{max_retries + 1} (timeout={timeout_seconds}s) - correlation_id={correlation_id}")
                start_time = time_module.time()
                
                # Execute with timeout
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(func(), timeout=timeout_seconds)
                else:
                    # For sync functions, run in executor with timeout
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, func),
                        timeout=timeout_seconds
                    )
                
                duration_ms = (time_module.time() - start_time) * 1000
                logger.info(f"✅ Attempt {attempt + 1} succeeded in {duration_ms:.0f}ms - correlation_id={correlation_id}")
                return result
                
            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Operation timed out after {timeout_seconds}s")
                logger.warning(f"⏱️ Attempt {attempt + 1} timed out - correlation_id={correlation_id}")
                
                if attempt < max_retries:
                    backoff_ms = 200 * (2 ** attempt)  # 200ms, 400ms, 800ms...
                    logger.info(f"🔄 Retrying in {backoff_ms}ms (exponential backoff) - correlation_id={correlation_id}")
                    await asyncio.sleep(backoff_ms / 1000)
                else:
                    logger.error(f"❌ All {max_retries + 1} attempts timed out - correlation_id={correlation_id}")
                    raise last_error
                    
            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ Attempt {attempt + 1} failed: {e} - correlation_id={correlation_id}")
                
                # Check if error is transient
                if self._is_transient_error(e) and attempt < max_retries:
                    backoff_ms = 200 * (2 ** attempt)  # 200ms, 400ms, 600ms (capped at 600ms per requirements)
                    if backoff_ms > 600:
                        backoff_ms = 600
                    logger.info(f"🔄 Retrying transient error in {backoff_ms}ms - correlation_id={correlation_id}")
                    await asyncio.sleep(backoff_ms / 1000)
                else:
                    # Non-transient error or out of retries
                    logger.error(f"❌ Failed after {attempt + 1} attempts: {e} - correlation_id={correlation_id}")
                    raise e
        
        # Should not reach here, but just in case
        raise last_error or Exception("Unknown error in _execute_with_timeout")
    
    def _final_read_verify(
        self,
        appointment_id: Optional[str],
        patient_id: str,
        doctor_id: str,
        date: str,
        time: str,
        correlation_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Final read verification: Query DB to confirm appointment exists with correct status.
        
        This is the CRITICAL final step - never confirm without this verification.
        
        Args:
            appointment_id: Appointment ID from create call (if available)
            patient_id: Patient ID
            doctor_id: Doctor ID
            date: Appointment date (YYYY-MM-DD)
            time: Appointment time (HH:MM)
            correlation_id: Session ID for logging
        
        Returns:
            Appointment dict if found and verified, None otherwise
        """
        correlation_id = correlation_id or "unknown"
        
        logger.info(f"🔍 FINAL READ VERIFY: correlation_id={correlation_id}")
        logger.info(f"   appointment_id={appointment_id}, patient_id={patient_id}, doctor_id={doctor_id}")
        logger.info(f"   date={date}, time={time}")
        
        # Try by appointment_id first (fastest)
        if appointment_id:
            try:
                appointment = self.db_client.get_appointment_by_id(appointment_id)
                if appointment:
                    logger.info(f"✅ Found appointment by ID: {appointment_id}")
                    # Verify details match
                    if self._verify_appointment_details(appointment, patient_id, doctor_id, date, time):
                        # Check status is confirmed/scheduled/pending
                        status = appointment.get('status', '')
                        if status in ['scheduled', 'confirmed', 'pending']:
                            logger.info(f"✅ FINAL READ VERIFY PASSED: Status={status}")
                            return appointment
                        else:
                            logger.error(f"❌ FINAL READ VERIFY FAILED: Invalid status={status}")
                            return None
            except Exception as e:
                logger.warning(f"⚠️ Failed to query by appointment_id: {e}")
        
        # Fallback: Query by patient + doctor + date + time
        try:
            logger.info(f"🔍 Fallback: Querying by patient+doctor+date+time")
            appointments = self.db_client.get_patient_appointments(patient_id)
            
            if appointments:
                for apt in appointments:
                    apt_date = str(apt.get('appointment_date', ''))
                    if 'T' in apt_date:
                        apt_date = apt_date.split('T')[0]
                    
                    apt_time = str(apt.get('start_time', ''))[:5]
                    expected_time_short = time[:5] if len(time) >= 5 else time
                    
                    if (apt.get('doctor_id') == doctor_id and
                        date in apt_date and
                        expected_time_short == apt_time):
                        logger.info(f"✅ Found appointment by query: {apt.get('id')}")
                        # Verify details
                        if self._verify_appointment_details(apt, patient_id, doctor_id, date, time):
                            status = apt.get('status', '')
                            if status in ['scheduled', 'confirmed', 'pending']:
                                logger.info(f"✅ FINAL READ VERIFY PASSED: Status={status}")
                                return apt
                            else:
                                logger.error(f"❌ FINAL READ VERIFY FAILED: Invalid status={status}")
                                return None
        except Exception as e:
            logger.error(f"❌ Failed to query by patient+doctor+date+time: {e}")
        
        logger.error(f"❌ FINAL READ VERIFY FAILED: Appointment not found in DB")
        return None
    
    # ========== DB VERIFICATION METHODS ==========
    # These methods ensure we never send false confirmations to users
    
    def _verify_appointment_in_db(
        self,
        appointment_id: str,
        max_retries: int = 3,
        retry_delay_ms: int = 500
    ) -> Optional[Dict[str, Any]]:
        """
        Verify appointment exists in database with retries.
        
        This is a CRITICAL safety check to prevent false confirmations.
        Never trust API response alone - always verify DB state.
        
        Args:
            appointment_id: The appointment ID to verify
            max_retries: Maximum retry attempts (default 3)
            retry_delay_ms: Delay between retries in milliseconds (default 500)
        
        Returns:
            Appointment dict if found and valid, None otherwise
        """
        import time as time_module
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🔍 Verification attempt {attempt + 1}/{max_retries}: Querying DB for appointment {appointment_id}")
                
                # Query DB for the appointment
                appointment = self.db_client.get_appointment_by_id(appointment_id)
                
                if appointment:
                    logger.info(f"✅ Verification attempt {attempt + 1}: Found appointment {appointment_id}")
                    logger.info(f"   Status: {appointment.get('status')}")
                    logger.info(f"   Patient: {appointment.get('patient_id')}")
                    logger.info(f"   Doctor: {appointment.get('doctor_id')}")
                    return appointment
                
                # Not found - wait and retry
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ Verification attempt {attempt + 1}: Appointment not found, retrying in {retry_delay_ms}ms...")
                    time_module.sleep(retry_delay_ms / 1000)
                
            except Exception as e:
                logger.error(f"❌ Verification attempt {attempt + 1} failed with error: {e}")
                if attempt < max_retries - 1:
                    time_module.sleep(retry_delay_ms / 1000)
        
        # All retries exhausted
        logger.error(f"❌ VERIFICATION FAILED: Appointment {appointment_id} not found after {max_retries} attempts")
        return None

    def _verify_appointment_details(
        self,
        appointment: Dict[str, Any],
        expected_patient_id: str,
        expected_doctor_id: str,
        expected_date: str,
        expected_time: str
    ) -> bool:
        """
        Verify appointment details match what was requested.
        
        This prevents scenarios where DB returns a different appointment
        or where data corruption occurs.
        
        Args:
            appointment: The appointment dict from DB
            expected_patient_id: Expected patient ID
            expected_doctor_id: Expected doctor ID
            expected_date: Expected date (YYYY-MM-DD)
            expected_time: Expected time (HH:MM)
        
        Returns:
            True if all details match, False otherwise
        """
        try:
            # Check patient
            actual_patient_id = appointment.get('patient_id')
            if actual_patient_id != expected_patient_id:
                logger.error(f"❌ Patient ID mismatch: expected={expected_patient_id}, actual={actual_patient_id}")
                return False
            
            # Check doctor
            actual_doctor_id = appointment.get('doctor_id')
            if actual_doctor_id != expected_doctor_id:
                logger.error(f"❌ Doctor ID mismatch: expected={expected_doctor_id}, actual={actual_doctor_id}")
                return False
            
            # Check date (handle different formats)
            actual_date = str(appointment.get('appointment_date', ''))
            # Extract just the date part if it includes time
            if 'T' in actual_date:
                actual_date = actual_date.split('T')[0]
            
            if expected_date not in actual_date and actual_date not in expected_date:
                logger.error(f"❌ Date mismatch: expected={expected_date}, actual={actual_date}")
                return False
            
            # Check time (compare HH:MM part)
            actual_time = str(appointment.get('start_time', ''))
            expected_time_short = expected_time[:5] if len(expected_time) >= 5 else expected_time
            actual_time_short = actual_time[:5] if len(actual_time) >= 5 else actual_time
            
            if expected_time_short != actual_time_short:
                logger.error(f"❌ Time mismatch: expected={expected_time_short}, actual={actual_time_short}")
                return False
            
            # Check status (should be scheduled, confirmed, or pending)
            status = appointment.get('status', '')
            valid_statuses = ['scheduled', 'confirmed', 'pending']
            if status not in valid_statuses:
                logger.error(f"❌ Invalid status: {status} (expected one of {valid_statuses})")
                return False
            
            logger.info(f"✅ All appointment details verified successfully")
            logger.info(f"   Patient: {actual_patient_id} ✓")
            logger.info(f"   Doctor: {actual_doctor_id} ✓")
            logger.info(f"   Date: {actual_date} ✓")
            logger.info(f"   Time: {actual_time_short} ✓")
            logger.info(f"   Status: {status} ✓")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error verifying appointment details: {e}")
            return False

    def _log_booking_metrics(
        self,
        appointment_inserted: bool,
        db_row_id: Optional[str],
        verification_latency_ms: float,
        scheduling_latency_ms: float,
        dashboard_visible: bool,
        request_id: Optional[str],
        mismatch_detected: bool = False,
        error_code: Optional[str] = None
    ):
        """
        Log structured metrics for appointment booking.
        
        These metrics are used for monitoring and alerting on false confirmations.
        
        Args:
            appointment_inserted: Whether appointment was inserted in DB
            db_row_id: The appointment ID if inserted
            verification_latency_ms: Time taken for DB verification
            scheduling_latency_ms: Time taken for API scheduling call
            dashboard_visible: Whether appointment is visible in dashboard
            request_id: Idempotency request ID
            mismatch_detected: Whether a false confirmation was detected
            error_code: Error code if any
        """
        import json
        from datetime import datetime as dt
        
        metrics = {
            "appointment_inserted": appointment_inserted,
            "db_row_id": db_row_id,
            "verification_latency_ms": round(verification_latency_ms, 2),
            "scheduling_call_latency_ms": round(scheduling_latency_ms, 2),
            "dashboard_visible": dashboard_visible,
            "request_id": request_id,
            "mismatch_detected": mismatch_detected,
            "last_error_code": error_code,
            "timestamp": dt.now().isoformat()
        }
        
        # Log structured metrics
        logger.info(f"📊 METRICS: {json.dumps(metrics)}")
        
        # Log threshold alerts
        if verification_latency_ms > 15000:
            logger.error(f"🚨 THRESHOLD CRITICAL: verification_latency_ms ({verification_latency_ms:.0f}) > 15000ms")
        elif verification_latency_ms > 5000:
            logger.warning(f"⚠️ THRESHOLD WARN: verification_latency_ms ({verification_latency_ms:.0f}) > 5000ms")
        
        # Log mismatch alert
        if mismatch_detected:
            logger.error(f"🚨 MISMATCH DETECTED: API returned success but DB verification failed!")
            logger.error(f"   appointment_id={db_row_id}, request_id={request_id}")
        
        return metrics
