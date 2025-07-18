#!/usr/bin/env python3
"""
Test script to verify the resizable UI functionality
"""

import pygame
import sys
import os

# Initialize Pygame
pygame.init()

# Test the UILayout class from the main application
class UILayout:
    def __init__(self, width, height, target_count):
        self.min_width = 800
        self.min_height = 500
        self.width = max(self.min_width, width)
        self.height = max(self.min_height, height)
        self.target_count = target_count
        self.update_layout()
    
    def update_layout(self):
        # Ensure minimum content fits - scale based on window size
        base_scale = min(self.width / 1200, self.height / 600)  # Reference size
        
        # Header area for stats - responsive but with minimum
        self.header_height = max(60, int(80 * base_scale))
        
        # Button area - scales with window but maintains usability
        self.button_height = max(35, min(100, int(60 * base_scale)))
        
        # Auto mode button - better responsive sizing
        auto_button_width = max(80, min(180, int(self.width * 0.12)))
        auto_button_height = max(25, min(50, int(self.height * 0.06)))
        self.auto_rect = pygame.Rect(self.width - auto_button_width - 10, 5, 
                                   auto_button_width, auto_button_height)
        
        # Column layout for images
        self.col_width = self.width // 3
        
        # Calculate available space more carefully
        reserved_space = self.header_height + self.button_height + 40  # margins
        self.image_area_height = max(200, self.height - reserved_space)
        
        # Target directory buttons - ensure they fit
        self.button_width = max(60, self.width // max(1, self.target_count))
        self.buttons = []
        for idx in range(self.target_count):
            rect = pygame.Rect(idx * self.button_width, 
                             self.height - self.button_height, 
                             self.button_width, 
                             self.button_height)
            self.buttons.append(rect)
        
        # Image slot positioning - more responsive
        self.image_start_y = self.header_height + 10
        
        # Calculate image area more intelligently
        available_width = self.col_width - 20  # margins
        text_space = int(80 * base_scale)  # space for text below image
        available_height = self.image_area_height - text_space
        
        self.max_image_width = max(100, available_width * 0.9)
        self.max_image_height = max(80, available_height * 0.85)
        
        # Text areas within each slot
        self.text_area_height = text_space

# Dynamic font scaling function
def get_scaled_fonts(width, height):
    base_font_size = max(16, min(36, width // 40))
    small_font_size = max(12, min(24, width // 60))
    return pygame.font.SysFont(None, base_font_size), pygame.font.SysFont(None, small_font_size)

def main():
    # Start with a resizable window
    screen_width = 1200
    screen_height = 600
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
    pygame.display.set_caption("Resizable UI Test")
    
    # Test with 3 target directories
    target_count = 3
    ui_layout = UILayout(screen_width, screen_height, target_count)
    font, small_font = get_scaled_fonts(screen_width, screen_height)
    
    # Test data
    test_buttons = [
        (ui_layout.buttons[0], "folder1", "Folder 1 (1)"),
        (ui_layout.buttons[1], "folder2", "Folder 2 (2)"),
        (ui_layout.buttons[2], "folder3", "Folder 3 (3)")
    ]
    
    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Handle window resizing
            if event.type == pygame.VIDEORESIZE:
                screen_width, screen_height = event.w, event.h
                screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
                
                # Update UI layout
                ui_layout = UILayout(screen_width, screen_height, target_count)
                
                # Update fonts
                font, small_font = get_scaled_fonts(screen_width, screen_height)
                
                # Rebuild test buttons
                test_buttons = [
                    (ui_layout.buttons[0], "folder1", "Folder 1 (1)"),
                    (ui_layout.buttons[1], "folder2", "Folder 2 (2)"),
                    (ui_layout.buttons[2], "folder3", "Folder 3 (3)")
                ]
                
                print(f"Window resized to {screen_width}x{screen_height}")
                print(f"Header height: {ui_layout.header_height}")
                print(f"Button height: {ui_layout.button_height}")
                print(f"Image area height: {ui_layout.image_area_height}")
                print(f"Column width: {ui_layout.col_width}")
                print(f"Auto button: {ui_layout.auto_rect}")
                print("---")
        
        # Draw everything
        screen.fill((0, 0, 0))
        
        # Draw header area background
        pygame.draw.rect(screen, (20, 20, 40), (0, 0, screen_width, ui_layout.header_height))
        
        # Draw header info
        header_text = f"Window: {screen_width}x{screen_height} | Scale: {min(screen_width/1200, screen_height/600):.2f}"
        header_surface = small_font.render(header_text, True, (255, 255, 255))
        screen.blit(header_surface, (10, 10))
        
        # Draw layout info
        layout_text = f"Header: {ui_layout.header_height}px | Image Area: {ui_layout.image_area_height}px | Button: {ui_layout.button_height}px"
        layout_surface = small_font.render(layout_text, True, (200, 200, 200))
        screen.blit(layout_surface, (10, 30))
        
        # Draw column dividers
        for i in range(1, 3):
            x = i * ui_layout.col_width
            pygame.draw.line(screen, (100, 100, 100), (x, ui_layout.header_height), 
                           (x, ui_layout.height - ui_layout.button_height), 2)
        
        # Draw image areas with boundaries
        for slot in range(3):
            x = slot * ui_layout.col_width + 10
            y = ui_layout.image_start_y
            w = ui_layout.col_width - 20
            h = ui_layout.image_area_height - 20
            
            # Draw main slot border
            pygame.draw.rect(screen, (50, 50, 150), (x, y, w, h), 2)
            
            # Draw max image size indicator
            max_img_x = x + (w - ui_layout.max_image_width) // 2
            max_img_y = y + 10
            pygame.draw.rect(screen, (0, 150, 0), 
                           (max_img_x, max_img_y, ui_layout.max_image_width, ui_layout.max_image_height), 1)
            
            # Draw text area
            text_y = max_img_y + ui_layout.max_image_height + 5
            text_height = ui_layout.text_area_height
            pygame.draw.rect(screen, (150, 100, 0), 
                           (x + 5, text_y, w - 10, text_height), 1)
            
            # Draw slot info
            slot_text = font.render(f"Slot {slot + 1}", True, (200, 200, 200))
            text_x = x + (w - slot_text.get_width()) // 2
            text_y_center = y + h // 2 - slot_text.get_height() // 2
            screen.blit(slot_text, (text_x, text_y_center))
            
            # Show dimensions
            dim_text = small_font.render(f"Img: {int(ui_layout.max_image_width)}x{int(ui_layout.max_image_height)}", 
                                       True, (100, 255, 100))
            screen.blit(dim_text, (x + 5, max_img_y + ui_layout.max_image_height + 25))
            
            # Show available space
            avail_text = small_font.render(f"Text: {ui_layout.text_area_height}px", True, (255, 200, 0))
            screen.blit(avail_text, (x + 5, max_img_y + ui_layout.max_image_height + 45))
        
        # Draw bottom button area background
        pygame.draw.rect(screen, (40, 20, 20), (0, ui_layout.height - ui_layout.button_height, 
                                               screen_width, ui_layout.button_height))
        
        # Draw target buttons
        for idx, (rect, _, label) in enumerate(test_buttons):
            # Color based on fit
            button_color = (0, 200, 0) if rect.width >= 60 else (200, 100, 0)
            pygame.draw.rect(screen, button_color, rect)
            
            # Scale button text to fit
            button_text = font.render(label, True, (0, 0, 0))
            if button_text.get_width() > rect.width - 10:
                # Try smaller font
                smaller_font = pygame.font.SysFont(None, max(12, font.get_height() - 4))
                button_text = smaller_font.render(label, True, (0, 0, 0))
                if button_text.get_width() > rect.width - 10:
                    # Truncate if still too wide
                    max_chars = max(3, len(label) * (rect.width - 10) // button_text.get_width())
                    truncated = label[:max_chars-3] + "..." if max_chars < len(label) else label
                    button_text = smaller_font.render(truncated, True, (0, 0, 0))
            
            text_rect = button_text.get_rect(center=rect.center)
            screen.blit(button_text, text_rect)
            
            # Show button width
            width_text = small_font.render(f"{rect.width}px", True, (255, 255, 255))
            screen.blit(width_text, (rect.x + 2, rect.y - 15))
        
        # Draw auto mode button
        auto_color = (0, 0, 255) if ui_layout.auto_rect.width >= 80 else (255, 100, 0)
        pygame.draw.rect(screen, auto_color, ui_layout.auto_rect)
        auto_text = small_font.render("Auto Mode", True, (255, 255, 255))
        if auto_text.get_width() > ui_layout.auto_rect.width - 10:
            auto_text = small_font.render("Auto", True, (255, 255, 255))
        auto_text_rect = auto_text.get_rect(center=ui_layout.auto_rect.center)
        screen.blit(auto_text, auto_text_rect)
        
        # Draw warning if window is too small
        if screen_width < 800 or screen_height < 500:
            warning_text = font.render("WARNING: Window too small for optimal display!", True, (255, 0, 0))
            screen.blit(warning_text, (screen_width//2 - warning_text.get_width()//2, screen_height//2))
        
        # Draw content fit indicator
        fit_status = "✓ All content fits" if (ui_layout.image_area_height > 200 and 
                                            ui_layout.max_image_height > 80 and 
                                            ui_layout.text_area_height > 60) else "⚠ Content may be cramped"
        fit_color = (0, 255, 0) if "✓" in fit_status else (255, 100, 0)
        fit_text = small_font.render(fit_status, True, fit_color)
        screen.blit(fit_text, (10, screen_height - ui_layout.button_height - 20))
        
        pygame.display.flip()
    
    pygame.quit()
    print("Test completed successfully!")

if __name__ == "__main__":
    main()
